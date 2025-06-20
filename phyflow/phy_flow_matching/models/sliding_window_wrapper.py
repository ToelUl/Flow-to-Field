import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Union


class GenerativeSlidingWindowWrapper(nn.Module):
    """
    Sliding-window inference wrapper for generative models (e.g., Flow Matching, Diffusion).

    This wrapper handles inputs that include a time step `t` and multiple conditioning variables.
    It automatically identifies and processes spatial conditions (e.g., masks) by patchifying them,
    and passes global conditions (e.g., embeddings) directly to the core model.

    Attributes:
        core_model (nn.Module): Underlying generative model. Expected signature:
            ``forward(self, x: torch.Tensor, t: torch.Tensor, **conditions) -> torch.Tensor``.
        patch_size (Tuple[int, int]): Height and width of each patch (patch_h, patch_w).
        stride (Tuple[int, int]): Vertical and horizontal stride for sliding window (stride_y, stride_x).
        padding_mode (str): Padding mode for torch.nn.functional.pad. Defaults to 'constant'.
        model_batch_size (int): Batch size used when processing patches. Defaults to 64.
    """

    def __init__(
        self,
        core_model: nn.Module,
        patch_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding_mode: str = 'constant',
        model_batch_size: int = 64
    ) -> None:
        """
        Initializes the GenerativeSlidingWindowWrapper.

        Args:
            core_model: Core generative model implementing the inference logic.
            patch_size: Tuple of (patch_height, patch_width).
            stride: Tuple of (stride_y, stride_x).
            padding_mode: Padding mode for input tensors. Defaults to 'constant'.
            model_batch_size: Number of patches to process at once. Defaults to 64.
        """
        super().__init__()
        self.core_model = core_model
        self.patch_size = patch_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.model_batch_size = model_batch_size

    @staticmethod
    def _patchify(
        image: torch.Tensor,
        patch_size: Tuple[int, int],
        stride: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Splits a single padded image or spatial condition into patches.

        Args:
            image: Tensor of shape (B, C, H, W).
            patch_size: Tuple (patch_h, patch_w).
            stride: Tuple (stride_y, stride_x).

        Returns:
            Tensor of shape (num_patches, C, patch_h, patch_w).
        """
        b, c, h, w = image.shape
        patch_h, patch_w = patch_size
        stride_y, stride_x = stride

        patches_h = image.unfold(2, patch_h, stride_y)
        patches_hw = patches_h.unfold(3, patch_w, stride_x)
        patches_hw = patches_hw.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches_hw.view(-1, c, patch_h, patch_w)
        return patches

    @staticmethod
    def _unpatchify(
        patches: torch.Tensor,
        output_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        stride: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Reconstructs a full image from patchified outputs, averaging overlaps.

        Args:
            patches: Tensor of shape (N, C, patch_h, patch_w).
            output_size: Tuple (height, width) of the padded image.
            patch_size: Tuple (patch_h, patch_w).
            stride: Tuple (stride_y, stride_x).

        Returns:
            Reconstructed image tensor of shape (1, C, H, W).
        """
        n, c, patch_h, patch_w = patches.shape
        # Flatten each patch to a vector
        patches_reshaped = patches.view(n, -1).T.unsqueeze(0)
        summed_image = F.fold(
            patches_reshaped,
            output_size=output_size,
            kernel_size=patch_size,
            stride=stride
        )

        # Count overlaps for averaging
        one_patches = torch.ones_like(patches)
        one_reshaped = one_patches.view(n, -1).T.unsqueeze(0)
        overlap_count = F.fold(
            one_reshaped,
            output_size=output_size,
            kernel_size=patch_size,
            stride=stride
        )

        # Avoid division by zero
        final_image = summed_image / overlap_count.clamp(min=1.0)
        return final_image

    def forward(
        self,
        x_batch: torch.Tensor,
        t: torch.Tensor,
        **conditions: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Performs conditional sliding-window inference on a batch of images.

        Args:
            x_batch: Input tensor of shape (B, C, H, W).
            t: Time-step tensor of shape (B,) or a scalar.
            **conditions: Additional keyword conditions.
                - Spatial conditions: Tensor with same H, W dimensions as x_batch; will be patchified.
                - Global conditions: Other tensors or data passed directly.

        Returns:
            Tensor of shape (B, C, H, W) with the aggregated output.
        """
        self.core_model.eval()

        b, c, img_h, img_w = x_batch.shape
        patch_h, patch_w = self.patch_size
        stride_y, stride_x = self.stride

        # 1. Separate spatial and global conditions
        spatial_conditions: Dict[str, torch.Tensor] = {}
        global_conditions: Dict[str, Any] = {}
        for key, value in conditions.items():
            if (
                isinstance(value, torch.Tensor)
                and value.dim() == 4
                and value.shape[2] == img_h
                and value.shape[3] == img_w
            ):
                spatial_conditions[key] = value
            else:
                global_conditions[key] = value

        # 2. Compute padding amounts and pad inputs
        pad_h = (stride_y - (img_h - patch_h) % stride_y) % stride_y
        pad_w = (stride_x - (img_w - patch_w) % stride_x) % stride_x
        padding = (0, pad_w, 0, pad_h)

        padded_x = F.pad(x_batch, padding, mode=self.padding_mode)
        padded_h, padded_w = padded_x.shape[2], padded_x.shape[3]

        padded_spatial_conditions = {
            key: F.pad(val, padding, mode=self.padding_mode)
            for key, val in spatial_conditions.items()
        }

        # 3. Process each item in the batch
        outputs = []
        for i in range(b):
            single_x = padded_x[i].unsqueeze(0)
            single_t = t[i] if t.dim() > 0 else t

            single_spatial = {
                key: val[i].unsqueeze(0)
                for key, val in padded_spatial_conditions.items()
            }
            single_global = {
                key: val[i] if isinstance(val, torch.Tensor) else val
                for key, val in global_conditions.items()
            }

            # 4. Extract patches
            x_patches = self._patchify(single_x, self.patch_size, self.stride)
            cond_patches = {
                key: self._patchify(val, self.patch_size, self.stride)
                for key, val in single_spatial.items()
            }

            # 5. Run core model on patches in mini-batches
            processed = []
            with torch.no_grad():
                for j in range(0, x_patches.size(0), self.model_batch_size):
                    x_mb = x_patches[j : j + self.model_batch_size]
                    kwargs = {key: val[j : j + self.model_batch_size] for key, val in cond_patches.items()}
                    kwargs.update(single_global)
                    out_mb = self.core_model(x_mb, t=single_t, **kwargs)
                    processed.append(out_mb)

            all_patches = torch.cat(processed, dim=0)

            # 6. Reconstruct padded output
            recon_padded = self._unpatchify(
                all_patches,
                output_size=(padded_h, padded_w),
                patch_size=self.patch_size,
                stride=self.stride
            )
            outputs.append(recon_padded)

        batch_recon = torch.cat(outputs, dim=0)

        # 7. Remove padding to restore original size
        final_output = batch_recon[:, :, :img_h, :img_w]
        return final_output
