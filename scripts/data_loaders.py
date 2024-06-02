import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from scripts.scale_transform_magnitude import compute_stm


def process_ensemble(row, audio_file_path):
    y, sr = librosa.load(audio_file_path, offset=row.start, duration=row.duration)
    label = row.label
    return y, sr, label


def load_malian_jembe_dataset():
    """
    Audio segments are loaded based on time-stamps contained in the annotations.

    Returns
    -------
    features_mj: list
        List of feature vectors.
    labels_mj: list
        List of labels.
    hover_data_mj: pandas.DataFrame
        DataFrame with hover data.
    """
    # Define paths to the dataset
    mj_media = Path("../datasets/MJ/Media")
    mj_annotations = Path("../datasets/MJ/Annotations")

    # Initialize lists for feature extraction
    features_mj = []
    labels_mj = []
    pattern, tradition, instrument = [], [], []

    # Define column names for annotation files
    columns = ["category", "start", "end", "duration", "label"]

    # Define a list for duration values
    durations = []

    # Iterate over annotation folders
    for parent_folder in mj_annotations.iterdir():
        if parent_folder.is_dir():
            # Define the path to the audio files
            audio_files_path = mj_media / parent_folder.name

            # Iterate over annotation files
            for annot in parent_folder.glob("*Annotation.csv"):
                print(f"Processing annotation file: {annot.name}")

                # Read the annotation CSV file
                annot_df = pd.read_csv(
                    annot,
                    header=None,
                    names=columns,
                    dtype={"start": float, "end": float, "duration": float},
                )

                # Filter the rows for category "MUSIC_FORM" and duration > 10
                annot_df = annot_df[
                    (annot_df["category"] == "MUSIC_FORM") & (annot_df["duration"] > 10)
                ]

                # Extract the ensemble number from the annotation file name
                ensemble_num = annot.name.rstrip("Annotation.csv")

                # Iterate over audio files
                for audio_file in audio_files_path.glob("*.wav"):
                    if ensemble_num in audio_file.name:
                        print(f"Processing audio file: {audio_file}")

                        # Iterate over rows in the annotation data frame
                        for row in annot_df.itertuples(index=False):
                            # Append the duration to the durations list
                            durations.append(row.duration)

                            # Process the ensemble and extract the feature
                            y, sr, label = process_ensemble(
                                row=row, audio_file_path=audio_file
                            )

                            # Append the label and feature to the corresponding lists
                            labels_mj.append(f"{label}_{audio_file.stem}")
                            stm = compute_stm(y=y, sr=sr)[:100]
                            features_mj.append(stm)

                            # Split the label and append parts to the corresponding lists
                            parts = label.split("_")
                            if len(parts) >= 3:
                                pattern.append(parts[0])
                                tradition.append(parts[2])
                                instrument.append(parts[4])

    # Create the hover data DataFrame
    hover_data_mj = pd.DataFrame(
        {"pattern": pattern, "instrument": instrument, "tradition": tradition}
    )
    hover_data_mj["label"] = tradition
    hover_data_mj = hover_data_mj.reset_index(drop=True)

    # Print the duration median
    print(f"Features shape: {len(features_mj)} -- labels shape: {len(labels_mj)}")
    print(f"Duration median: {np.median(durations)} seconds")

    return features_mj, labels_mj, hover_data_mj


def load_candombe_dataset():
    """
    Each wav file is segmented into 20 second long non-overlapping segments.

    Returns:
        features_candombe (list): A list of feature vectors.
        labels_candombe (list): A list of corresponding labels.
        hover_data_candombe (DataFrame): A DataFrame with additional information for hovering in the visualization.
    """

    # Path to the Candombe dataset
    candombe_media = Path("../datasets/candombe/Media")

    # Lists to store the features, labels, and instrument information
    labels_candombe = []
    features_candombe = []

    # Iterate over all wav files in the dataset
    for file in candombe_media.rglob("*.wav"):
        label = file.stem.split("_")[-1]

        # Skip the stereo mix files
        if label == "Stereo":
            continue

        print(f"Processing file: {file.name}")

        # Load the audio file
        y, sr = librosa.load(file)

        # Segment length in seconds
        segment_length = 20

        # Iterate over the segments
        for start in range(0, len(y), segment_length * sr):
            # Compute the STM for the segment
            segment = y[start : start + segment_length * sr]
            stm = compute_stm(y=segment, sr=sr)[:100]

            # Append the feature and label
            features_candombe.append(stm)
            labels_candombe.append(f"{label}_{start // (segment_length * sr)}")

    # Split the label and append parts to the corresponding lists
    pattern, instrument = [], []
    for label in labels_candombe:
        parts = label.split("_")
        pattern.append(parts[1])
        instrument.append(parts[0])

    # Create the hover data DataFrame
    hover_data_candombe = pd.DataFrame({"pattern": pattern, "instrument": instrument})
    hover_data_candombe["tradition"] = "Candombe"
    hover_data_candombe["label"] = instrument
    hover_data_candombe = hover_data_candombe.reset_index(drop=True)

    print(
        f"Features shape: {len(features_candombe)} -- labels shape: {len(labels_candombe)}"
    )

    return features_candombe, labels_candombe, hover_data_candombe
