"""
Download utilities for pyflowreg demo data.

Provides centralized download functionality for all demo datasets.
"""

import urllib.request
from pathlib import Path
from typing import Optional, Union

# Public dictionary mapping demo names to their download URLs
DEMO_DATA_URLS = {
    "jupiter.tiff": "https://drive.usercontent.google.com/download?id=12lEhahzKtOZsFgxLzwxnT8JsVBErvzJH&export=download&authuser=0",
    "synth_frames.h5": "https://drive.usercontent.google.com/download?id=10YxHVSdnz0L4WMLR0eIHH6bMxaojpVdY&export=download&authuser=0",
    "injection.tiff": "https://drive.usercontent.google.com/download?id=15WiDimFrWheSLq6rYvdWMthiuMUb0DR6&export=download&authuser=0",
}


def download_data(url: str, filename: str, output_folder: Union[str, Path]) -> Path:
    """
    Download a file from a URL to a specified folder.

    Parameters
    ----------
    url : str
        The URL to download from.
    filename : str
        The name to save the file as.
    output_folder : Union[str, Path]
        The folder to save the file in.

    Returns
    -------
    Path
        The path to the downloaded file.

    Raises
    ------
    urllib.error.URLError
        If the download fails.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    file_path = output_folder / filename

    if file_path.exists():
        print(f"File already exists: {file_path}")
        return file_path

    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, file_path)
        print(f"Downloaded to {file_path}")
    except Exception as e:
        print(f"Download failed: {e}")
        raise

    return file_path


def download_demo_data(
    demo_name: str, output_folder: Optional[Union[str, Path]] = None
) -> Path:
    """
    Download demo data by name using predefined URLs.

    Parameters
    ----------
    demo_name : str
        The name of the demo data file to download. Must be one of the keys in DEMO_DATA_URLS.
    output_folder : Optional[Union[str, Path]]
        The folder to save the file in. If None, uses 'data' folder relative to project root.

    Returns
    -------
    Path
        The path to the downloaded file.

    Raises
    ------
    ValueError
        If the demo_name is not found in DEMO_DATA_URLS.
    urllib.error.URLError
        If the download fails.

    Examples
    --------
    >>> # Download jupiter demo data
    >>> jupiter_path = download_demo_data("jupiter.tiff")

    >>> # Download synthetic evaluation data
    >>> synth_path = download_demo_data("synth_frames.h5")
    """
    if demo_name not in DEMO_DATA_URLS:
        available = ", ".join(DEMO_DATA_URLS.keys())
        raise ValueError(f"Unknown demo data: {demo_name}. Available: {available}")

    if output_folder is None:
        # Use the data folder at project root
        # Go up from src/pyflowreg/util to project root, then to data/
        current_file = Path(__file__)
        # __file__ is in src/pyflowreg/util/, so go up 3 levels to get to project root
        project_root = current_file.parent.parent.parent.parent
        output_folder = project_root / "data"

    url = DEMO_DATA_URLS[demo_name]
    return download_data(url, demo_name, output_folder)


if __name__ == "__main__":
    # Quick test of download functionality
    print("Testing download utilities...")

    # Test downloading jupiter data
    try:
        jupiter_path = download_demo_data("jupiter.tiff")
        print(f"✓ Jupiter data: {jupiter_path}")
        print(f"  File size: {jupiter_path.stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"✗ Failed to download jupiter data: {e}")

    # Test downloading synth data
    try:
        synth_path = download_demo_data("synth_frames.h5")
        print(f"✓ Synth data: {synth_path}")
        print(f"  File size: {synth_path.stat().st_size / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"✗ Failed to download synth data: {e}")

    print("\nAll available demo data:")
    for name, url in DEMO_DATA_URLS.items():
        print(f"  - {name}")
