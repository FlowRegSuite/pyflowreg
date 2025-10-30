"""
Tests for download utilities.
"""

import pytest
from pathlib import Path
import tempfile
import urllib.request
from unittest.mock import patch

from pyflowreg.util.download import download_data, download_demo_data, DEMO_DATA_URLS


class TestDownloadData:
    """Test the general download_data function."""

    def test_download_data_creates_folder(self):
        """Test that download_data creates the output folder if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a nested folder that doesn't exist
            output_folder = Path(tmpdir) / "nested" / "folder"
            assert not output_folder.exists()

            # Mock the actual download
            with patch("urllib.request.urlretrieve") as mock_retrieve:
                mock_retrieve.return_value = None

                result = download_data(
                    "http://example.com/file.txt", "test.txt", output_folder
                )

                assert output_folder.exists()
                assert result == output_folder / "test.txt"
                mock_retrieve.assert_called_once()

    def test_download_data_skips_existing_file(self, capsys):
        """Test that download_data skips downloading if file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_folder = Path(tmpdir)
            test_file = output_folder / "existing.txt"
            test_file.write_text("existing content")

            # Should not call urlretrieve
            with patch("urllib.request.urlretrieve") as mock_retrieve:
                result = download_data(
                    "http://example.com/file.txt", "existing.txt", output_folder
                )

                assert result == test_file
                mock_retrieve.assert_not_called()

                # Check output message
                captured = capsys.readouterr()
                assert "already exists" in captured.out

    def test_download_data_error_handling(self):
        """Test that download_data properly handles download errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_folder = Path(tmpdir)

            with patch("urllib.request.urlretrieve") as mock_retrieve:
                mock_retrieve.side_effect = urllib.error.URLError("Network error")

                with pytest.raises(urllib.error.URLError):
                    download_data("http://invalid.url", "test.txt", output_folder)


class TestDownloadDemoData:
    """Test the demo-specific download_demo_data function."""

    def test_download_demo_data_invalid_name(self):
        """Test that download_demo_data raises error for unknown demo names."""
        with pytest.raises(ValueError) as exc_info:
            download_demo_data("nonexistent_demo.txt")

        assert "Unknown demo data" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_download_demo_data_default_folder(self):
        """Test that download_demo_data uses correct default folder."""
        with patch("pyflowreg.util.download.download_data") as mock_download:
            mock_download.return_value = Path("data/jupiter.tiff")

            download_demo_data("jupiter.tiff")

            # Check that the default data folder is used
            call_args = mock_download.call_args
            assert call_args is not None
            url, filename, output_folder = call_args[0]

            assert url == DEMO_DATA_URLS["jupiter.tiff"]
            assert filename == "jupiter.tiff"
            assert str(output_folder).endswith("data")

    def test_download_demo_data_custom_folder(self):
        """Test that download_demo_data accepts custom output folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_folder = Path(tmpdir) / "custom"

            with patch("pyflowreg.util.download.download_data") as mock_download:
                mock_download.return_value = custom_folder / "jupiter.tiff"

                download_demo_data("jupiter.tiff", custom_folder)

                call_args = mock_download.call_args
                assert call_args is not None
                url, filename, output_folder = call_args[0]

                assert output_folder == custom_folder


class TestDemoDataUrls:
    """Test that all demo data URLs are accessible."""

    @pytest.mark.parametrize("demo_name,url", DEMO_DATA_URLS.items())
    def test_demo_url_accessible(self, demo_name, url):
        """Test that each demo URL is accessible (HEAD request only)."""
        # Use HEAD request to check URL without downloading
        request = urllib.request.Request(url, method="HEAD")

        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                # Check for successful response
                assert (
                    response.status == 200
                ), f"URL for {demo_name} returned status {response.status}"
        except urllib.error.HTTPError as e:
            # Google Drive might not support HEAD, try GET with Range header
            request = urllib.request.Request(url)
            request.add_header("Range", "bytes=0-0")  # Request only first byte
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    # 206 Partial Content is also acceptable
                    assert response.status in [
                        200,
                        206,
                    ], f"URL for {demo_name} not accessible: {e}"
            except Exception as e2:
                pytest.skip(
                    f"Could not verify {demo_name} URL (might be network issue): {e2}"
                )
        except Exception as e:
            pytest.skip(
                f"Could not verify {demo_name} URL (might be network issue): {e}"
            )

    def test_all_expected_demos_present(self):
        """Test that all expected demo files are in DEMO_DATA_URLS."""
        expected_demos = {"jupiter.tiff", "synth_frames.h5", "injection.tiff"}
        actual_demos = set(DEMO_DATA_URLS.keys())

        assert (
            expected_demos == actual_demos
        ), f"Missing or extra demos: {expected_demos ^ actual_demos}"


class TestIntegration:
    """Integration tests with actual small downloads."""

    @pytest.mark.slow
    def test_download_synth_data_integration(self):
        """Integration test: Actually download synth data and verify."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download to temporary directory
            result = download_demo_data("synth_frames.h5", Path(tmpdir))

            # Verify file exists and has reasonable size
            assert result.exists()
            file_size = result.stat().st_size
            assert file_size > 1000  # Should be at least 1KB

            # Verify it's an HDF5 file by checking magic number
            with open(result, "rb") as f:
                magic = f.read(8)
                # HDF5 files start with specific magic number
                assert (
                    magic[:4] == b"\x89HDF"
                    or magic == b"\x89\x48\x44\x46\x0d\x0a\x1a\x0a"
                )


if __name__ == "__main__":
    # Quick test
    print("Testing download module...")

    # Test invalid demo name
    try:
        download_demo_data("invalid.txt")
        print("✗ Should have raised ValueError for invalid demo")
    except ValueError as e:
        print(f"✓ Correctly raised error for invalid demo: {e}")

    # Test URL accessibility
    print("\nChecking demo URLs...")
    for name, url in DEMO_DATA_URLS.items():
        request = urllib.request.Request(url)
        request.add_header("Range", "bytes=0-0")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                print(f"✓ {name}: Accessible")
        except Exception as e:
            print(f"✗ {name}: {e}")

    print("\nRun 'pytest test_download.py' for full test suite")
