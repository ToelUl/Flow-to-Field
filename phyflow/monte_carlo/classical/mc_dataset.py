import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Union, Optional

class MCDataset(Dataset):
    """Dataset for Monte Carlo spin configurations with temperature and (optional) system size labels.

    This dataset processes spin configuration data and corresponding labels consisting of
    temperature and optionally system size. The spin configurations are reshaped and the labels are expanded
    such that each spin configuration sample is paired with a label vector.
    If system size is included, the label is [T, L]; otherwise, the label is the scalar T.

    Attributes:
        data (torch.Tensor): Reshaped spin configuration data with shape (B * C*N, 1, L, L).
        labels (torch.Tensor): Expanded labels. If include_system_size is True,
                               shape is (B * C*N, 2); if False, shape is (B * C*N,).
        include_system_size (bool): Whether system size is included in the labels.
    """

    def __init__(self, data: torch.Tensor, labels: torch.Tensor,
                 include_system_size: bool = False, rad_to_vector: bool = False,
                 transform: Optional[transforms.Compose] = None ) -> None:
        """Initializes the MonteCarloDataset.

        Args:
            data (torch.Tensor): Tensor of spin configurations with original shape (B, C*N, L, L),
                where B is the batch size, C*N is the combined channel dimension, and L is the spatial dimension.
            labels (torch.Tensor): Tensor of labels.
                - If include_system_size is True, shape should be (B, 2), where the first column is the
                  temperature T and the second column is the system size L.
                - If include_system_size is False, shape should be (B,), containing only the temperature T.
            include_system_size (bool, optional): Whether to include system size in the label. Defaults to False.
            rad_to_vector (bool, optional): If True, convert radial coordinates to vector form.
            transform (Optional[transforms.Compose], optional): Optional transform to be applied to the data.

        Raises:
            AssertionError: If `data` does not have 4 dimensions.
            AssertionError: If `labels` dimensions or size do not match the `include_system_size` setting.
            AssertionError: If the batch size (B) of `data` and `labels` do not match.
        """
        # Validate input dimensions
        assert len(data.shape) == 4, "Expected data shape to be (B, C*N, L, L)"
        assert data.shape[0] == labels.shape[0], "The batch dimension (B) of data and labels must match"
        self.include_system_size = include_system_size

        if self.include_system_size:
            assert len(labels.shape) == 2 and labels.shape[1] == 2, \
                "When include_system_size=True, expected labels shape to be (B, 2)"
            label_dim = 2
        else:
            # Allow shape (B,) or (B, 1), but standardize to (B,)
            if len(labels.shape) == 2 and labels.shape[1] == 1:
                labels = labels.squeeze(1) # Convert from (B, 1) to (B,)
            assert len(labels.shape) == 1, \
                "When include_system_size=False, expected labels shape to be (B,) or (B, 1)"
            label_dim = None # For scalar labels, no specific dimension needed internally

        batch_size, channels_times_n, l_dim, _ = data.shape

        # Reshape data from (B, C*N, L, L) to (B * C*N, 1, L, L)
        self.data = data.reshape(batch_size * channels_times_n, 1, l_dim, l_dim)
        if rad_to_vector:
            # Convert radial coordinates to vector form
            self.data = torch.cat([torch.cos(self.data), torch.sin(self.data)], dim=1)

        # Expand labels
        if self.include_system_size:
            # Repeat each (T, L) label channels_times_n times to get shape (B, channels_times_n, 2),
            # then flatten the first two dimensions to form a tensor of shape (B * C*N, 2)
            self.labels = labels.unsqueeze(1).expand(batch_size, channels_times_n, label_dim).reshape(-1, label_dim)
        else:
            # Repeat each T label channels_times_n times to get shape (B, channels_times_n),
            # then flatten the first two dimensions to form a tensor of shape (B * C*N,)
            self.labels = labels.unsqueeze(1).expand(batch_size, channels_times_n).reshape(-1)

        # Apply optional transformation to the data
        self.transform = transform


    def __len__(self) -> int:
        """Returns the total number of samples.

        Returns:
            int: The total number of samples, equal to B * C*N.
        """
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, torch.Tensor]]:
        """Retrieves the spin configuration and corresponding label for the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, Union[torch.Tensor, torch.Tensor]]: A tuple containing:
                - A tensor of shape (1, L, L) representing the spin configuration.
                - The label tensor:
                    - If include_system_size was True, shape is (2,), representing [temperature, system size].
                    - If include_system_size was False, shape is () (scalar tensor), representing the temperature.
        """
        x = self.data[idx]  # Spin configuration, shape (1, L, L)
        if self.transform:
            x = self.transform(x)
        y = self.labels[idx]  # Label vector or scalar
        return x, y

# ====================================================
# Simple verification of the dataset functionality
# ====================================================
if __name__ == "__main__":
    # Create dummy data: batch size B=2, channels_times_n=3, spatial dimension L=4
    dummy_data = torch.randn(2, 3, 4, 4)

    # --- Case 1: Including system size (default behavior) ---
    print("--- Case 1: Including system size ---")
    # Create dummy labels with temperature and system size
    dummy_labels_with_L = torch.tensor([[0.893, 4.], [2.269, 4.]])  # Shape: (2, 2)

    # Instantiate the dataset (using default include_system_size=True)
    dataset_with_L = MCDataset(dummy_data, dummy_labels_with_L, include_system_size=True)

    # Print dataset length
    print("Total samples:", len(dataset_with_L))  # Expected output: 2 * 3 = 6

    # Retrieve a sample and print its details
    sample_data, sample_label = dataset_with_L[0]
    print("Sample data shape:", sample_data.shape)   # Expected shape: (1, 4, 4)
    print("Sample label:", sample_label)             # Expected shape: (2,) -> tensor([0.8930, 4.0000])
    print("Sample label shape:", sample_label.shape) # Expected torch.Size([2])

    sample_data_last, sample_label_last = dataset_with_L[-1]
    print("Last sample label:", sample_label_last)    # Expected shape: (2,) -> tensor([2.2690, 4.0000])

    print("\n")

    # --- Case 2: Not including system size ---
    print("--- Case 2: Not including system size ---")
    # Create dummy labels with only temperature
    dummy_labels_only_T = torch.tensor([0.893, 2.269]) # Shape: (2,)

    # Instantiate the dataset, explicitly setting include_system_size=False
    dataset_only_T = MCDataset(dummy_data, dummy_labels_only_T, include_system_size=False)

    # Print dataset length
    print("Total samples:", len(dataset_only_T))  # Expected output: 2 * 3 = 6

    # Retrieve a sample and print its details
    sample_data, sample_label = dataset_only_T[0]
    print("Sample data shape:", sample_data.shape)   # Expected shape: (1, 4, 4)
    print("Sample label:", sample_label)             # Expected shape: () -> tensor(0.8930)
    print("Sample label shape:", sample_label.shape) # Expected torch.Size([])

    sample_data_last, sample_label_last = dataset_only_T[-1]
    print("Last sample label:", sample_label_last)    # Expected shape: () -> tensor(2.2690)

    # --- Case 3: Label shape is (B, 1) and not including system size ---
    print("\n--- Case 3: Label shape is (B, 1) and not including system size ---")
    dummy_labels_only_T_unsqueeze = torch.tensor([[0.893], [2.269]]) # Shape: (2, 1)
    dataset_only_T_unsqueezed = MCDataset(dummy_data, dummy_labels_only_T_unsqueeze, include_system_size=False)
    print("Total samples:", len(dataset_only_T_unsqueezed)) # Expected output: 6
    sample_data, sample_label = dataset_only_T_unsqueezed[0]
    print("Sample data shape:", sample_data.shape)   # Expected shape: (1, 4, 4)
    print("Sample label:", sample_label)             # Expected shape: () -> tensor(0.8930)
    print("Sample label shape:", sample_label.shape) # Expected torch.Size([])

    # --- Testing Assertions ---
    print("\n--- Testing Assertions ---")
    try:
        # Label shape error (expected (B, 2), but got (B,))
        error_dataset = MCDataset(dummy_data, dummy_labels_only_T, include_system_size=True)
    except AssertionError as e:
        print(f"Successfully caught assertion error (include_system_size=True): {e}")

    try:
        # Label shape error (expected (B,) or (B, 1), but got (B, 2))
        error_dataset = MCDataset(dummy_data, dummy_labels_with_L, include_system_size=False)
    except AssertionError as e:
        print(f"Successfully caught assertion error (include_system_size=False): {e}")

    try:
        # Batch size mismatch
        dummy_labels_wrong_batch = torch.tensor([1.0, 2.0, 3.0]) # B=3, while data is B=2
        error_dataset = MCDataset(dummy_data, dummy_labels_wrong_batch, include_system_size=False)
    except AssertionError as e:
        print(f"Successfully caught assertion error (Batch size mismatch): {e}")