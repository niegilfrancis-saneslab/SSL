import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def get_args() -> tuple[Path, list[Path]]:
    """Parse command line arguments to get the output path and input paths.

    Returns:
        tuple[Path, list[Path]]: The output path and the input paths
    """

    parser = argparse.ArgumentParser(
        description="Merge multiple HDF5 files into a single HDF5 file."
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="The path to the output HDF5 file.",
    )
    parser.add_argument(
        "input_paths",
        type=Path,
        nargs="+",
        help="The paths to the input HDF5 files.",
    )
    args = parser.parse_args()
    return args.output_path, args.input_paths


def merge(input_paths: list[Path], output_path: Path) -> None:
    """Merge multiple HDF5 files into a single HDF5 file.

    Args:
        input_paths (list[Path]): The paths to the input HDF5 files.
        output_path (Path): The path to the output HDF5 file.

    Raises:
        ValueError: If all input files do not have the same number of channels.
    """

    # Loop through all inputs to get the size of the output dataset

    num_channels = None
    location_shape = None
    trackid_shape = None
    audio_lengths = []
    num_locations = 0

    attrs = {}

    for input_path in input_paths:
        with h5py.File(input_path, "r") as ctx_out:
            if num_channels is None:
                num_channels = ctx_out["audio"].shape[1]
            if location_shape is None:
                location_shape = ctx_out["locations"].shape[1:]
            if trackid_shape is None:
                trackid_shape = ctx_out["track_id"].shape[1:]

            if ctx_out["audio"].shape[1] != num_channels:
                raise ValueError(
                    "All input files must have the same number of channels."
                )
            if ctx_out["locations"].shape[1:] != location_shape:
                raise ValueError(
                    "All input files must have the same number of animals, nodes, and dimensions"
                )
            if ctx_out["track_id"].shape[1:] != trackid_shape:
                raise ValueError(
                    "All input files must have the same number of animals"
                )
            
            

            audio_lengths.extend(np.diff(ctx_out["length_idx"][:]))
            num_locations += ctx_out["locations"].shape[0]

            for k, v in ctx_out.attrs.items():
                attrs[k] = v

    length_idx = np.cumsum([0] + audio_lengths)

    with h5py.File(output_path, "w") as ctx_out:
        audio_dset = ctx_out.create_dataset(
            "audio",
            shape=(length_idx[-1], num_channels),
            chunks=(1024, num_channels),
            dtype="f",
        )
        locations_dset = ctx_out.create_dataset(
            "locations",
            shape=(num_locations, *location_shape),
            dtype="f",
        )
        labels_dset = ctx_out.create_dataset(
            "labels",
            shape=(num_locations,),
            dtype = "f",
        )
        trackid_dset = ctx_out.create_dataset(
            "track_id",
            shape=(num_locations, *trackid_shape),
            dtype = "S40"
        )
        ctx_out.create_dataset("length_idx", data=length_idx)

        orig_fnames = np.array(list(map(lambda p: bytes(p.name, "utf-8"), input_paths)))
        ctx_out.create_dataset("orig_filenames", data=orig_fnames)
        orig_file_dset = ctx_out.create_dataset(
            "orig_file", shape=(num_locations,), dtype="i"
        )

        ctx_out.attrs.update(attrs)

        cur_audio_idx = 0
        cur_location_idx = 0


        for file_idx, in_f in tqdm(enumerate(input_paths), total=len(input_paths)):
            with h5py.File(in_f, "r") as ctx_in:
                audio_in = ctx_in["audio"]
                locations_in = ctx_in["locations"]
                labels_in = ctx_in["labels"]
                trackid_in = ctx_in["track_id"]

                audio_dset[cur_audio_idx : cur_audio_idx + audio_in.shape[0]] = audio_in
                locations_dset[
                    cur_location_idx : cur_location_idx + locations_in.shape[0]
                ] = locations_in
                orig_file_dset[
                    cur_location_idx : cur_location_idx + locations_in.shape[0]
                ] = file_idx

                labels_dset[
                    cur_location_idx : cur_location_idx + locations_in.shape[0]
                ] = labels_in

                trackid_dset[
                    cur_location_idx : cur_location_idx + locations_in.shape[0]
                ] = trackid_in


                cur_audio_idx += audio_in.shape[0]
                cur_location_idx += locations_in.shape[0]

                for key in ctx_in.keys():
                    if key in ("audio", "locations","track_id","labels"):
                        continue
                    if key in ctx_out:
                        continue
                    ctx_in.copy(key, ctx_out)

        for key in ctx_out.keys():
            print(key, ctx_out[key].shape)


if __name__ == "__main__":
    output_path, input_paths = get_args()
    input_paths.sort()
    merge(input_paths, output_path)
