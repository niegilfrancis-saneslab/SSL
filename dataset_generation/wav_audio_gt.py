from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile


def read_wavs_from_dir(experiment_dir: Path) -> tuple[float, np.ndarray]:
    """Reads all the WAV files in the experiment directory and stores them in RAM
    as a multi-channel audio array.

    Args:
        experiment_dir (Path): Path to the experiment directory containing WAV files. The wav files are
        expected to be named in the format 'channel_#_*.wav', where # is the 0-indexed channel number.

    Returns:
        float: Sample rate of the audio files
        np.ndarray: A multi-channel audio array with shape (num_samples, num_channels)
    """

    sorter = lambda p: int(p.stem.split("_")[1])
    wav_paths = list(experiment_dir.glob("channel_*_*.wav"))
    if not wav_paths:
        raise ValueError(
            f"No WAV files found in {experiment_dir} matching the pattern 'channel_*_*.wav'."
        )
    wav_paths.sort(key=sorter)
    num_channels = len(wav_paths)
    sr, audio = wavfile.read(wav_paths[0])
    data = np.empty((audio.shape[0], num_channels), dtype=audio.dtype)
    for i, wav_path in enumerate(wav_paths):
        _, audio = wavfile.read(wav_path)
        if len(audio) != data.shape[0]:
            raise ValueError(
                f"Audio length mismatch: {wav_path} has {len(audio)} samples, expected {data.shape[0]}."
            )
        data[:, i] = audio

    return sr, data


def process_annotations(annotation_path: Path) -> np.ndarray:
    """Reads the annotation file and returns a structured array with vocalization annotations.

    Args:
        annotation_path (Path): Path to the annotation CSV file.

    Returns:
        np.ndarray: A structured array with columns for start time, end time, and label.
    """

    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found at {annotation_path}.")

    df = pd.read_csv(annotation_path)
    # columns: name,start_seconds,stop_seconds,channel
    good_rows = df["name"].isin(["vox", "noise"])
    df = df[good_rows]

    # Verify that there are no overlapping annatations
    start_times = df["start_seconds"].to_numpy()
    end_times = df["stop_seconds"].to_numpy()
    labels = df['label'].to_numpy()
    
    # gaps = start_times[1:] - end_times[:-1]
    #if gaps.min() < 0:
        #pass
        # raise ValueError(
        #     f"Overlapping annotations detected in {annotation_path}. Please ensure that the annotations do not overlap."
        # )

    segments = np.stack([start_times, end_times], axis=-1)
    labels = labels[~np.isnan(segments).any(axis=1)]

    segments = segments[~np.isnan(segments).any(axis=1), :]
    
    return segments, labels


class ExperimentAudio:
    def __init__(self, experiment_dir: Path):
        """A class which reads audio files and vocalization annotations from the experiment directory and
        facilitates access to each vocalization instance.

        Args:
            experiment_dir (Path): Path to the experiment directory containing WAV files and annotation file.
        """
        self.experiment_dir = experiment_dir
        annotation_path = next(experiment_dir.glob("*annotations*gt*.csv"), None)
        if annotation_path is None or (not annotation_path.exists()):
            raise FileNotFoundError(f"Annotation file not found at {annotation_path}.")
        self.sample_rate, self.full_audio = read_wavs_from_dir(experiment_dir)
        self.segments_sec, self.labels = process_annotations(annotation_path)


    def __len__(self) -> int:
        """Returns the number of vocalizations."""
        return len(self.segments_sec)

    def __getitem__(self, idx: int) -> np.ndarray:
        """Returns the vocalization of the given index as a multi-channel audio array."""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for vocalizations.")
        start, end = self.segments_sec[idx]
        start_sample = int(start * self.sample_rate)
        end_sample = int(end * self.sample_rate)
        return self.full_audio[start_sample:end_sample, :]

    def __iter__(self):
        """Returns an iterator over the vocalizations."""
        for idx in range(len(self)):
            yield self[idx]


if __name__ == "__main__":
    sample_experiment_dir = Path("/home/atanelus/box_data/experiment_135/idx_0")
    audio = ExperimentAudio(sample_experiment_dir)
