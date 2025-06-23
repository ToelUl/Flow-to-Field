import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple

class GenerativeSlidingWindowWrapper(nn.Module):
    """Sliding-window inference wrapper for generative models.

    This wrapper modifies a generative model to perform inference on arbitrarily
    large images by breaking them into overlapping patches, processing them in
    batches, and seamlessly reassembling the results. It is designed to handle
    inputs that include a time step `t` and multiple conditioning variables,
    both spatial (like masks) and global (like embeddings).

    The key improvement is that it processes the entire batch of images in
    parallel, avoiding explicit loops over individual images and leveraging
    batched tensor operations for significant speedup.

    Attributes:
        core_model (nn.Module): The underlying generative model. Its forward pass
            is expected to be `forward(x, t, **conditions)`.
        patch_size (Tuple[int, int]): The height and width of each patch.
        stride (Tuple[int, int]): The vertical and horizontal stride for the
            sliding window.
        padding_mode (str): The padding mode used by `torch.nn.functional.pad`.
            Defaults to 'circular'.
        model_batch_size (int): The maximum number of patches to process at once
            in the `core_model` to manage memory usage. Defaults to 64.
    """

    def __init__(
        self,
        core_model: nn.Module,
        patch_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding_mode: str = 'circular',
        model_batch_size: int = 64
    ) -> None:
        """Initializes the GenerativeSlidingWindowWrapper.

        Args:
            core_model: The core generative model to be wrapped.
            patch_size: A tuple (patch_height, patch_width).
            stride: A tuple (stride_y, stride_x).
            padding_mode: Padding mode for input tensors. Defaults to 'constant'.
            model_batch_size: Number of patches to process simultaneously in the
                core model. Defaults to 64.
        """
        super().__init__()
        self.core_model = core_model
        self.patch_size = patch_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.model_batch_size = model_batch_size

    @staticmethod
    def _patchify(
        images: torch.Tensor,
        patch_size: Tuple[int, int],
        stride: Tuple[int, int]
    ) -> torch.Tensor:
        """Splits a batch of padded images into patches.

        Args:
            images: A batch of images of shape (B, C, H, W).
            patch_size: A tuple (patch_h, patch_w).
            stride: A tuple (stride_y, stride_x).

        Returns:
            A tensor of patches with shape
            (B * num_patches_per_image, C, patch_h, patch_w).
        """
        b, c, h, w = images.shape
        patch_h, patch_w = patch_size
        stride_y, stride_x = stride

        # Use unfold to create sliding window views of the tensor
        patches_h = images.unfold(2, patch_h, stride_y)
        patches_hw = patches_h.unfold(3, patch_w, stride_x)

        # Reshape to get the final patch tensor
        # (B, C, num_patches_y, num_patches_x, patch_h, patch_w)
        # -> (B, num_patches_y, num_patches_x, C, patch_h, patch_w)
        patches_hw = patches_hw.permute(0, 2, 3, 1, 4, 5).contiguous()

        # (B, num_patches_y, num_patches_x, C, patch_h, patch_w)
        # -> (B * num_patches_y * num_patches_x, C, patch_h, patch_w)
        patches = patches_hw.view(-1, c, patch_h, patch_w)
        return patches

    @staticmethod
    def _unpatchify(
        patches: torch.Tensor,
        batch_size: int,
        output_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        stride: Tuple[int, int]
    ) -> torch.Tensor:
        """Reconstructs a batch of images from patches, averaging overlaps.

        Args:
            patches: Tensor of shape (B * N, C, patch_h, patch_w), where N is
                the number of patches per image.
            batch_size: The original batch size (B).
            output_size: The tuple (height, width) of the padded target image.
            patch_size: The tuple (patch_h, patch_w).
            stride: The tuple (stride_y, stride_x).

        Returns:
            A reconstructed batch of images of shape (B, C, H, W).
        """
        num_total_patches, c, patch_h, patch_w = patches.shape
        num_patches_per_image = num_total_patches // batch_size

        # Reshape patches to be suitable for F.fold
        # (B * N, C, patch_h, patch_w) -> (B, N, C, patch_h, patch_w)
        patches_reshaped = patches.view(batch_size, num_patches_per_image, c, patch_h, patch_w)

        # (B, N, C, patch_h, patch_w) -> (B, N, C * patch_h * patch_w)
        patches_reshaped = patches_reshaped.view(batch_size, num_patches_per_image, -1)

        # (B, N, C * patch_h * patch_w) -> (B, C * patch_h * patch_w, N)
        patches_for_fold = patches_reshaped.permute(0, 2, 1)

        # Use F.fold for batched reconstruction
        summed_images = F.fold(
            patches_for_fold,
            output_size=output_size,
            kernel_size=patch_size,
            stride=stride
        )

        # Create a tensor of ones to count overlaps
        one_patches = torch.ones_like(patches_for_fold)
        overlap_count = F.fold(
            one_patches,
            output_size=output_size,
            kernel_size=patch_size,
            stride=stride
        )

        # Average the results by dividing by the overlap count
        final_images = summed_images / overlap_count.clamp(min=1.0)
        return final_images

    def forward(
        self,
        x_batch: torch.Tensor,
        t: torch.Tensor,
        **conditions: Dict[str, Any]
    ) -> torch.Tensor:
        """Performs batched sliding-window inference.

        Args:
            x_batch: The input tensor batch of shape (B, C, H, W).
            t: The time-step tensor of shape (B,) or a scalar.
            **conditions: Additional keyword conditions.
                - Spatial conditions: Tensors with the same H, W as x_batch.
                - Global conditions: Other tensors (e.g., embeddings).

        Returns:
            The aggregated output tensor of shape (B, C, H, W).
        """
        self.core_model.eval()

        b, c, img_h, img_w = x_batch.shape
        patch_h, patch_w = self.patch_size
        stride_y, stride_x = self.stride

        # 1. Separate spatial and global conditions
        spatial_conditions, global_conditions = {}, {}
        for key, value in conditions.items():
            is_spatial = (
                isinstance(value, torch.Tensor) and
                value.dim() == 4 and
                value.shape[2] == img_h and
                value.shape[3] == img_w
            )
            if is_spatial:
                spatial_conditions[key] = value
            else:
                global_conditions[key] = value

        # 2. Compute padding and pad all spatial tensors
        pad_h = (stride_y - (img_h - patch_h) % stride_y) % stride_y
        pad_w = (stride_x - (img_w - patch_w) % stride_x) % stride_x
        padding = (0, pad_w, 0, pad_h)

        padded_x = F.pad(x_batch, padding, mode=self.padding_mode)
        padded_h, padded_w = padded_x.shape[2], padded_x.shape[3]

        padded_spatial_conditions = {
            key: F.pad(val, padding, mode=self.padding_mode)
            for key, val in spatial_conditions.items()
        }

        # 3. Batch-patchify all spatial tensors
        x_patches = self._patchify(padded_x, self.patch_size, self.stride)
        cond_patches = {
            key: self._patchify(val, self.patch_size, self.stride)
            for key, val in padded_spatial_conditions.items()
        }

        num_patches_per_image = x_patches.shape[0] // b

        # 4. Prepare conditions for the core model
        # Expand time and global conditions to match the number of patches
        if t.dim() == 0: # Scalar t
            t_expanded = t.expand(x_patches.shape[0])
        else: # Batched t
            t_expanded = t.repeat_interleave(num_patches_per_image)

        expanded_global_conds = {}
        for key, val in global_conditions.items():
            if isinstance(val, torch.Tensor):
                # Assumes global conditions have a batch dimension
                expanded_global_conds[key] = val.repeat_interleave(num_patches_per_image, dim=0)
            else: # Non-tensor global conditions are duplicated for each patch
                expanded_global_conds[key] = val

        # 5. Run core model on all patches in mini-batches
        processed_patches = []
        with torch.no_grad():
            for i in range(0, x_patches.size(0), self.model_batch_size):
                end_idx = i + self.model_batch_size
                x_mb = x_patches[i:end_idx]
                t_mb = t_expanded[i:end_idx]

                # Combine all conditions for the mini-batch
                kwargs = {key: val[i:end_idx] for key, val in cond_patches.items()}
                kwargs.update({key: val[i:end_idx] for key, val in expanded_global_conds.items()})

                out_mb = self.core_model(x_mb, t=t_mb, **kwargs)
                processed_patches.append(out_mb)

        all_processed_patches = torch.cat(processed_patches, dim=0)

        # 6. Batch-unpatchify to reconstruct the full-sized batch
        batch_recon = self._unpatchify(
            all_processed_patches,
            batch_size=b,
            output_size=(padded_h, padded_w),
            patch_size=self.patch_size,
            stride=self.stride
        )

        # 7. Remove padding to restore original dimensions
        final_output = batch_recon[:, :, :img_h, :img_w]

        return final_output

# --- Test Suite ---
class IdentityModel(nn.Module):
    """
    A dummy identity model that simply returns its input tensor `x`.
    This is used for testing the wrapper to ensure that the process of
    patchifying and unpatchifying correctly reconstructs the original input.
    """
    def forward(self, x: torch.Tensor, t: torch.Tensor, **conditions) -> torch.Tensor:
        return x

def main():
    """
    Main function to run the test suite for the GenerativeSlidingWindowWrapper.
    """
    # 1. Test Configuration
    BATCH_SIZE = 4
    CHANNELS = 3
    IMG_H, IMG_W = 256, 384  # Use non-square images for robust testing
    PATCH_H, PATCH_W = 128, 128
    STRIDE_Y, STRIDE_X = 64, 64
    MODEL_BATCH_SIZE = 16  # Simulate processing 16 patches at a time
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("--- Test Configuration ---")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: ({IMG_H}, {IMG_W})")
    print(f"Patch Size: ({PATCH_H}, {PATCH_W})")
    print(f"Stride: ({STRIDE_Y}, {STRIDE_X})")
    print("-" * 26)

    # 2. Initialize Model and Wrapper
    core_model = IdentityModel().to(DEVICE)
    wrapper = GenerativeSlidingWindowWrapper(
        core_model=core_model,
        patch_size=(PATCH_H, PATCH_W),
        stride=(STRIDE_Y, STRIDE_X),
        model_batch_size=MODEL_BATCH_SIZE
    ).to(DEVICE)

    # 3. Prepare Test Data
    # Using random data for simulation
    x_batch = torch.randn(BATCH_SIZE, CHANNELS, IMG_H, IMG_W).to(DEVICE)
    t_batch = torch.linspace(0, 1, BATCH_SIZE).to(DEVICE)

    # Create a spatial condition (e.g., a mask) and a global condition (e.g., an embedding)
    spatial_condition = torch.randn(BATCH_SIZE, 1, IMG_H, IMG_W).to(DEVICE)
    global_condition = torch.randn(BATCH_SIZE, 128).to(DEVICE)

    conditions = {
        "mask": spatial_condition,
        "embedding": global_condition
    }

    print("\n--- Executing Sliding Window Inference ---")

    # 4. Run the Forward Pass
    output = wrapper(x_batch, t=t_batch, **conditions)

    print("Inference complete.")

    # 5. Verify the Results
    print("\n--- Verifying Results ---")
    print(f"Input tensor shape:  {x_batch.shape}")
    print(f"Output tensor shape: {output.shape}")

    # Verification 1: Check if output shape matches input shape
    assert x_batch.shape == output.shape, "Verification failed: Output shape does not match input shape!"
    print("✅ Verification successful: Output shape is correct.")

    # Verification 2: Check if output content matches input content.
    # Because the core model is an identity function, the output should be a
    # nearly perfect reconstruction of the input. We use `torch.allclose` to
    # account for potential minor floating-point errors from the division
    # during the averaging of overlapping patches.
    are_close = torch.allclose(x_batch, output, atol=1e-6)
    assert are_close, "Verification failed: Output content does not match input content!"
    print("✅ Verification successful: Output content is correct.")

if __name__ == '__main__':
    main()