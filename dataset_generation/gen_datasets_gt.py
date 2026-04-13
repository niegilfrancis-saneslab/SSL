import argparse
from collections import namedtuple
from pathlib import Path

import cv2
import h5py
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.patches import Polygon
from tqdm import tqdm

from csv_tracks_gt import extract_tracks
from wav_audio_gt import ExperimentAudio
import traceback

RAMDataset = namedtuple(
    "RAMDataset", ["audio", "locations","labels","track_id","node_names", "frame_indices", "metadata"]
)
animal_colors = ["#9DFFF9", "#FFA552", "#ED97B4", "#8CD747"]


def write_dataset_to_disk(output_path: Path, dset: RAMDataset):
    """Writes the RAMDataset to an HDF5 file.

    Args:
        output_path (Path): Path to the output HDF5 file.
        dset (RAMDataset): The dataset to write, containing audio, locations, and metadata.
    """
    length_idx = np.array([0] + [len(a) for a in dset.audio])
    length_idx = np.cumsum(length_idx)
    concat_audio = np.concatenate(dset.audio, axis=0)
    with h5py.File(output_path, "w") as ctx:
        ctx.create_dataset(
            "audio", data=concat_audio, chunks=(8192, concat_audio.shape[1])
        )
        ctx.create_dataset("locations", data=dset.locations)
        ctx.create_dataset("frame_indices", data=dset.frame_indices, dtype=np.int64)
        ctx.create_dataset("length_idx", data=length_idx, dtype=np.int64)
        ctx.create_dataset("node_names", data=np.array(dset.node_names, dtype="S"))
        ctx.create_dataset("labels", data=np.array(dset.labels, dtype=np.float16))
        ctx.create_dataset("track_id", data=np.array(dset.track_id, dtype="S40"))
        for key, value in dset.metadata.items():
            ctx.attrs[key] = value


def locate_experiment_dirs(base_dir: Path) -> list[Path]:
    """Locates all experiment directories under the base directory.

    Args:
        base_dir (Path): The base directory to search for experiment directories.

    Returns:
        list[Path]: A list of Paths to the experiment directories.
    """
    candidates = [p.parent for p in base_dir.glob("**/*.wav")]
    candidates = list(set(candidates))  # dedup
    return candidates


def get_video_info(video_path: Path) -> tuple[tuple[int, int], float]:
    """Gets the dimensions of the video file.

    Args:
        video_path (Path): Path to the video file.

    Returns:
        tuple[int, int]: Width and height of the video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    framerate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return (width, height), framerate


def make_dataset(exp_dir: Path, visualization_dir: Path | None = None) -> RAMDataset:
    """Creates a RAMDataset from the experiment directory.

    Args:
        exp_dir (Path): Path to the experiment directory containing audio files and video.

    Returns:
        RAMDataset: A dataset containing vocalizations, locations, and metadata.
    """
    exp_name = f"{exp_dir.parent.name}-{exp_dir.name}"
    track_path = next(exp_dir.glob("*center*corrected*.csv"), None)
    if track_path is None:
        raise FileNotFoundError(f"No tracks found in {exp_name}. Expected *.tracks.csv")
    try:
        audio_iter = ExperimentAudio(exp_dir)
    except Exception as e:
        return None
    video_file = next(exp_dir.glob("*.mp4"), None)
    if video_file is None:
        raise FileNotFoundError(f"No video file found in {exp_name}. Expected *.mp4")
    video_dims, video_framerate = get_video_info(video_file)
    animals = extract_tracks(pd.read_csv(track_path))


    vocalizations = []
    locations = []  # Will have shape (num_vocalizations, num_animals, num_nodes=2, 2)
    video_frame_indices = []
    track_id = []
    labels = []
    test = []
    for label, (start_sec, end_sec), vocalization in tqdm(
        zip(audio_iter.labels, audio_iter.segments_sec, iter(audio_iter)), total=len(audio_iter)
    ):
        mid_time_frames = int((start_sec + end_sec) / 2 * video_framerate)
        frame_tracks = np.stack([animal[mid_time_frames][0] for animal in animals])
        frame_track_ids = np.stack([animal[mid_time_frames][1] for animal in animals])

        if np.isnan(frame_tracks).any():
            continue  # Skip if any animal is not tracked at this time
            # This should only happen close to the start or end of the video
        locations.append(frame_tracks)
        vocalizations.append(vocalization.astype(np.float32))
        video_frame_indices.append(mid_time_frames)
        track_id.append(frame_track_ids)
        labels.append(label)
    locations = np.array(locations, dtype=np.float32)
    track_id = np.array(track_id, dtype="S40")
    labels = np.array(labels, dtype=np.float16)
    node_names = "nose", "head"
    shift = np.array(video_dims) / 2.0
    locations -= shift[
        None, None, None, :
    ]  # Move the origin to the center of the video
    metadata = {
        "experiment_name": exp_name,
        "sample_rate": audio_iter.sample_rate,
        "num_channels": vocalizations[0].shape[1],
        "num_animals": locations.shape[1],
        "arena_dims": video_dims,
    }

    dset = RAMDataset(
        audio=vocalizations,
        locations=locations,
        node_names=node_names,
        frame_indices=video_frame_indices,
        metadata=metadata,
        track_id = track_id,
        labels = labels
    )

    if visualization_dir is not None:
        print("Starting visualization...")
        visualize(dset, video_file, visualization_dir / f"{exp_name}.mp4")
    return dset


def visualize(dset: RAMDataset, video_path: Path, output_path: Path):
    """Generates a movie visualizing the dataset.

    Args:
        dset (RamDataset): dataset to visualize.
        output_path (Path): Path to save the output video. Should have .mp4 extension.
    """

    adims = dset.metadata["arena_dims"]
    fig_width_in = 16
    fig_height_in = 12
    dpi_physical = 100

    def gen_fig_image(
        audio: np.ndarray, tracks: np.ndarray, video_frame: np.ndarray
    ) -> np.ndarray:
        fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi_physical)
        spec = gs.GridSpec(4, 4, figure=fig, wspace=0, hspace=0)
        image_ax = fig.add_subplot(spec[1:3, 1:3])  # Video in the center
        audio_grids = [
            spec[0, 0],
            spec[0, 1],
            spec[0, 2],
            spec[0, 3],
            spec[1, 0],
            spec[1, 3],
            spec[2, 0],
            spec[2, 3],
            spec[3, 0],
            spec[3, 1],
            spec[3, 2],
            spec[3, 3],
        ]
        audio_axes = [fig.add_subplot(ax) for ax in audio_grids]
        spectrograms = []
        for n, ax in enumerate(audio_axes):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            _, _, _, im = ax.specgram(
                audio[:, n],
                NFFT=256,
                noverlap=128,
                cmap="magma",
            )
            spectrograms.append(im)

        min_c, max_c = (
            min([i.get_clim()[0] for i in spectrograms]),
            max([i.get_clim()[1] for i in spectrograms]),
        )
        for im in spectrograms:
            im.set_clim(min_c, max_c)

        image_ax.set_xticks([])
        image_ax.set_yticks([])
        image_ax.set_frame_on(False)
        image_ax.imshow(
            video_frame[..., ::-1],
            # y goes from pos to neg because the top of the video is low-y and matplotlib coords place the origin at the bottom left
            extent=[adims[0] / -2, adims[0] / 2, adims[1] / 2, adims[1] / -2],
        )
        for i, track in enumerate(tracks):
            gerbil_direction = track[0] - track[1]
            # Making the arrow head point correspond to the nose of the mouse is
            # more intuitive
            A = track[0]
            rot_90 = np.array([[0, -1], [1, 0]])
            ortho_direction = rot_90 @ gerbil_direction
            B = A - gerbil_direction * 1.0 + ortho_direction * 0.5
            C = A - gerbil_direction * 1.0 - ortho_direction * 0.5
            poly = Polygon(
                np.stack([A, B, C]),
                closed=True,
                color=animal_colors[i],
            )
            image_ax.add_patch(poly)
        true_width, true_height = fig.canvas.get_width_height(physical=True)
        fig.canvas.draw()
        buffer = fig.canvas.buffer_rgba()
        image = np.frombuffer(buffer, np.uint8).reshape(true_height, true_width, 4)[
            ..., 2::-1
        ]  # to BGR
        return image

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (int(adims[0]), int(adims[1])),
        isColor=True,
    )
    orig_video_reader = cv2.VideoCapture(str(video_path))
    if not orig_video_reader.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    def job_arguments():
        for audio, tracks, frame_idx in zip(
            dset.audio, dset.locations, dset.frame_indices
        ):
            orig_video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, video_frame = orig_video_reader.read()
            if not ret:
                break
            yield audio, tracks, video_frame

    frames = Parallel(n_jobs=-2, return_as="generator")(
        delayed(gen_fig_image)(audio, tracks, video_frame)
        for audio, tracks, video_frame in job_arguments()
    )
    for image in tqdm(frames, total=len(dset.audio), desc="Generating frames"):
        writer.write(image)
    writer.release()


if __name__ == "__main__":
    sample_base_directory = Path("/mnt/home/neurostatslab/ceph/saneslab_data/ssl_gt_data")
    exp_dirs = locate_experiment_dirs(sample_base_directory)
    exp_dirs.sort()
    output_dir = sample_base_directory / "compiled_datasets"
    output_dir.mkdir(exist_ok=True)
    visualization_dir = None
    # visualization_dir = sample_base_directory / "visualizations"
    # visualization_dir.mkdir(exist_ok=True)
    total_vox = 0
    for exp in exp_dirs:
        print(f"Processing experiment: {exp.name}")
        try:
            exp_name = f"{exp.parent.name}-{exp.name}"
            output_path = output_dir / f"{exp_name}.h5"
            if output_path.exists():
                continue
            dataset = make_dataset(exp, visualization_dir=visualization_dir)
            write_dataset_to_disk(output_path, dataset)
            print(
                f"Dataset written to {output_path}. Num vocalizations: {len(dataset.audio)}"
            )
            total_vox += len(dataset.audio)
        except Exception as hell:
            print(f"Failed to process {exp.name}: {hell}")
            # traceback.print_exc()
            # raise hell

    print(f"Total vocalizations processed: {total_vox}")
