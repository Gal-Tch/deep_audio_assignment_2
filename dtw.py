import numpy as np
from tqdm import tqdm

from data_acquisition import get_spectrogram_from_dataset

THRESHOLD = 200  # TODO: find the right Threshold
# SPEAKERS = ["male_1", "male_1", "female_1", "female_2"] # TODO: restore to real speakers
SPEAKERS = ["male_1", "male_1", "male_1", "male_1"]


def calculate_vectors_diff(vector_a, vector_b):
    return np.linalg.norm(vector_a - vector_b)


def dtw(mel_s_x: np.ndarray, mel_s_y: np.ndarray):
    x_len = mel_s_x.shape[1]
    y_len = mel_s_y.shape[1]
    dtw_cost = np.zeros(shape=[x_len, y_len])
    for i in range(x_len):
        for j in range(y_len):
            down_cost, left_cost, diag_cost = np.inf, np.inf, np.inf
            curr_cost = calculate_vectors_diff(mel_s_x[:, i], mel_s_y[:, j])
            if i == 0 and j == 0:
                dtw_cost[0][0] = curr_cost
                continue
            if i > 0:
                left_cost = dtw_cost[i - 1][j]
            if j > 0:
                down_cost = dtw_cost[i][j - 1]
            if i > 0 and j > 0:
                diag_cost = dtw_cost[i - 1][j - 1]
            dtw_cost[i][j] = curr_cost + np.min([left_cost, down_cost, diag_cost])
    return dtw_cost[-1][-1]


def compare_train_to_reference():
    reference_spectrograms = []
    training_distance_matrix = np.zeros(shape=[40, 11])

    for i in range(10):
        reference_spectrograms.append(get_spectrogram_from_dataset("representative", "reference", i))
    reference_spectrograms.append(get_spectrogram_from_dataset("representative", "reference", "random"))
    for speaker_num in range(len(SPEAKERS)):
        for digit in tqdm(range(10)):
            for reference_recording_num in range(len(reference_spectrograms)):
                curr_spectrogram = get_spectrogram_from_dataset("train", SPEAKERS[speaker_num], digit)
                curr_distance = dtw(curr_spectrogram, reference_spectrograms[reference_recording_num])
                training_distance_matrix[speaker_num * 10 + digit][reference_recording_num] = curr_distance
    return training_distance_matrix


def get_train_random_words_true_negatives():
    reference_spectrograms = []
    for i in range(10):
        reference_spectrograms.append(get_spectrogram_from_dataset("representative", "reference", str(i)))
    true_negatives = 0
    for speaker in SPEAKERS:
        random_spectrogram = get_spectrogram_from_dataset("train", speaker, "random")
        for reference_spectrogram in reference_spectrograms:
            score = dtw(random_spectrogram, reference_spectrogram)
            if score < THRESHOLD:
                break
        else:
            true_negatives += 1
    return true_negatives


def classify_training_set():
    training_distance_matrix = compare_train_to_reference()
    true_positives = 0
    true_negatives = get_train_random_words_true_negatives()
    minimum_digits = training_distance_matrix.argmin(axis=1)
    for i, min_digit in enumerate(minimum_digits):
        if min_digit == i % 10 and training_distance_matrix[i][min_digit] < THRESHOLD:
            true_positives += 1
    return (true_positives + true_negatives) / (len(minimum_digits) + 4)


if __name__ == '__main__':
    print(classify_training_set())
