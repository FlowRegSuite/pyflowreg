import numpy as np
import re
from collections import defaultdict


class DSFileReader:
    """
    A mixin class that provides a generic, multi-pass heuristic for finding
    the most likely data-containing datasets within a file.
    """

    def _find_datasets(self, datasets_with_info: list[tuple]) -> list:
        """
        Heuristic to find datasets based on a list of names and their shapes.

        Args:
            datasets_with_info (list[tuple]): A list where each element is a
                tuple containing (dataset_name: str, dataset_shape: tuple).

        Returns:
            A list of strings with the names of the selected datasets.
        """
        all_names = [info[0] for info in datasets_with_info]

        # --- Pass 1: Find datasets with channel conventions (e.g., 'ch1', 'channel_2') ---
        # This regex captures (prefix)(channel_word)(separator)(number)
        pattern = re.compile(
            r"^(.*?)((?:ch|channel|chan))([_.\s]*)(\d+)", re.IGNORECASE
        )

        channel_groups = defaultdict(list)
        for name in all_names:
            match = pattern.match(name)
            if match:
                prefix = match.group(1)
                channel_num = int(match.group(4))
                channel_groups[prefix].append((channel_num, name))

        if channel_groups:
            # Find the group with the most channels that also have consistent shapes
            # This is a crucial check to ensure we're getting a real channel group
            valid_groups = {}
            for prefix, channels in channel_groups.items():
                # Get shapes of all datasets in this group
                shapes = {
                    info[1]
                    for name in channels
                    for info in datasets_with_info
                    if info[0] == name[1]
                }
                if (
                    len(shapes) == 1
                ):  # All datasets in the group must have the same shape
                    valid_groups[prefix] = channels

            if valid_groups:
                best_prefix = max(valid_groups, key=lambda k: len(valid_groups[k]))
                sorted_channels = sorted(
                    valid_groups[best_prefix], key=lambda item: item[0]
                )
                print(
                    f"Heuristic Pass 1: Found channel group with prefix '{best_prefix}'."
                )
                return [name for num, name in sorted_channels]

        # --- Pass 2: Find datasets with common generic names ---
        common_names = ["mov", "data", "dataset"]
        for name in all_names:
            sanitized_name = name.lower().lstrip("/")
            if sanitized_name in common_names:
                print(f"Heuristic Pass 2: Found common dataset '{name}'.")
                return [name]

        # --- Pass 3: Fallback to guessing based on dimensions ---
        print("Heuristic Pass 1 & 2 failed. Falling back to dimension-based guessing.")

        candidate_shapes = defaultdict(list)
        for name, shape in datasets_with_info:
            if len(shape) in [3, 4]:
                candidate_shapes[shape].append(name)

        if candidate_shapes:
            best_shape = max(candidate_shapes, key=lambda s: np.prod(s))
            print(
                f"Warning: Guessing video data based on dimensions. "
                f"Selected {len(candidate_shapes[best_shape])} dataset(s) with shape {best_shape}."
            )
            return candidate_shapes[best_shape]

        return []


class DSFileWriter:
    """
    A mixin class that provides logic for generating dataset names for writers.
    This is a direct port of the DS_file_writer.m functionality.
    """

    def __init__(self, **kwargs):
        # Default dimension ordering for writers: (height, width, time)
        self.dimension_ordering = kwargs.get("dimension_ordering", (0, 1, 2))
        self.dataset_names = kwargs.get("dataset_names", None)

        # Sanitize dataset names by removing any leading slashes
        if self.dataset_names:
            if isinstance(self.dataset_names, list):
                self.dataset_names = [name.lstrip("/") for name in self.dataset_names]
            elif isinstance(self.dataset_names, str):
                self.dataset_names = self.dataset_names.lstrip("/")

    def get_ds_name(self, channel_id: int, n_channels: int) -> str:
        """
        Gets the dataset name for a specific channel.

        Args:
            channel_id (int): The 1-based index of the channel.
            n_channels (int): The total number of channels being written.

        Returns:
            The dataset name as a string.
        """
        if self.dataset_names:
            if isinstance(self.dataset_names, list):
                if len(self.dataset_names) != n_channels:
                    raise ValueError(
                        "The number of provided dataset names must match the number of channels."
                    )
                return self.dataset_names[channel_id - 1]

            # Handle string patterns like 'ch*_reg'
            if "*" in self.dataset_names:
                return self.dataset_names.replace("*", str(channel_id))

            # If it's a single name for a single channel, or a prefix for multiple
            if n_channels == 1:
                return self.dataset_names
            else:
                return f"{self.dataset_names}{channel_id}"
        else:
            # Default naming convention if none is provided
            return f"ch{channel_id}"
