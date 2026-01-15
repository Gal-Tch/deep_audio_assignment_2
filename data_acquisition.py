import os
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

RAW_DIR = 'raw_recordings'
DATASET_DIR = 'dataset'
TARGET_SR = 16000


def apply_agc(y):
    """
    Applies Automatic Gain Control (AGC) by normalizing the signal
    to have a maximum absolute amplitude of 1.0.
    """
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        return y / max_amp
    return y


def process_audio_file(file_path, output_path):
    """
    Loads an audio file, resamples it to 16kHz, and saves it as a WAV file.
    """
    try:
        y, sr = librosa.load(file_path, sr=TARGET_SR)
        sf.write(output_path, y, sr)
        print(f"Processed: {file_path} -> {output_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def create_dataset():
    """
    Walks through the raw_recordings directory, mirrors the structure in dataset,
    and converts audio files to wav format with resampling.
    """

    for root, dirs, files in os.walk(RAW_DIR):
        rel_path = os.path.relpath(root, RAW_DIR)
        target_dir = os.path.join(DATASET_DIR, rel_path)

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        for file in files:
            if file.endswith((".opus", "mov", "wav")):
                source_file = os.path.join(root, file)
                # Change extension to .wav
                target_filename = os.path.splitext(file)[0] + '.wav'
                target_file = os.path.join(target_dir, target_filename)

                process_audio_file(source_file, target_file)


TRAINING_SPEAKERS = ["male_1", "male_2", "female_1", "female_2"]
EVAL_SPEAKERS = ["male_3", "male_4", "female_3", "female_4"]


def compute_mel_spectrogram(audio_path):
    """
    Computes the Mel Spectrogram for a given audio file.
    Parameters:
        audio_path (str): Path to the .wav file.
    """
    y, sr = librosa.load(audio_path, sr=TARGET_SR)
    y = apply_agc(y)

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=int(0.025 * sr), hop_length=int(0.010 * sr),
                                                     n_mels=80)
    return mel_spectrogram


def get_spectrogram_from_dataset(speaker, digit):
    """
    Retrieves the Mel Spectrogram for a specific recording in the dataset.
    """
    if speaker in TRAINING_SPEAKERS:
        subset = "train"
    elif speaker in EVAL_SPEAKERS:
        subset = "eval"
    elif speaker == "reference":
        subset = "representative"
    else:
        raise ValueError(f"Unknown speaker: {speaker}")

    file_path = os.path.join(DATASET_DIR, subset, speaker, str(digit) + ".wav")
    return compute_mel_spectrogram(file_path)


def plot_mel_spectrogram(mel_spectrogram, sr, title, ax):
    """
    Plots a Mel Spectrogram on a given axis.
    """
    img = librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    ax.set_title(title)
    return img


def view_mel_spectrograms():
    """
    Analyzes and presents Mel Spectrograms to show:
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Within Speaker Differences (reference)')

    digits_within = [0, 5, 9]
    for i, digit in enumerate(digits_within):
        mel_spectrogram = get_spectrogram_from_dataset("reference", digit)
        plot_mel_spectrogram(mel_spectrogram, TARGET_SR, f'Digit: {digit}', axes[i])

    plt.tight_layout()
    plt.show()

    # 2. Differences across Digit Samples (Different Speakers/Genders)
    target_digit = 1
    speakers = ["female_1", "female_2", "male_1"]

    fig2, axes2 = plt.subplots(1, len(speakers), figsize=(5 * len(speakers), 5))
    fig2.suptitle(f'Differences across Speakers (Digit {target_digit})')

    for i, speaker in enumerate(speakers):
        mel_spectrogram = get_spectrogram_from_dataset(speaker, target_digit)
        plot_mel_spectrogram(mel_spectrogram, TARGET_SR, f'{speaker}', axes2[i])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    create_dataset()
    view_mel_spectrograms()
