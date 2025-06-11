
from pathlib import Path


def create_directory(root_path: str, folder_name: str) -> None:
    """Create a directory under the specified root path. Skip if it already exists.

    Args:
        root_path (str): Path to the existing root directory.
        folder_name (str): Name of the directory to create under the root.

    Raises:
        ValueError: If `root_path` does not exist or is not a directory.
        OSError: If the directory cannot be created due to filesystem permissions or other errors.
    """
    root = Path(root_path)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Root path does not exist or is not a directory: {root_path}")

    target_dir = root / folder_name
    try:
        # exist_ok=True ensures no error if the directory already exists
        target_dir.mkdir(parents=False, exist_ok=True)
    except OSError as e:
        raise OSError(f"Could not create directory '{folder_name}' under '{root_path}': {e}") from e
