import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from data_acquisition import get_spectrogram_from_dataset, TRAINING_SPEAKERS, EVAL_SPEAKERS

THRESHOLD = 2.5
SPEAKERS = EVAL_SPEAKERS
NORMALIZE = True


def calculate_vectors_diff(vector_a, vector_b):
    return np.linalg.norm(vector_a - vector_b)


def dtw(mel_s_x: np.ndarray, mel_s_y: np.ndarray, normalize: bool = False):
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

    if normalize:
        return dtw_cost[-1][-1] / (x_len + y_len)
    return dtw_cost[-1][-1]


def compare_to_reference():
    reference_spectrograms = []
    distance_matrix = np.zeros(shape=[40, 11])

    for i in range(10):
        reference_spectrograms.append(get_spectrogram_from_dataset("reference", i))
    reference_spectrograms.append(get_spectrogram_from_dataset("reference", "random"))
    for speaker_num in range(len(SPEAKERS)):
        for digit in tqdm(range(10)):
            for reference_recording_num in range(len(reference_spectrograms)):
                curr_spectrogram = get_spectrogram_from_dataset(SPEAKERS[speaker_num], digit)
                curr_distance = dtw(curr_spectrogram, reference_spectrograms[reference_recording_num],
                                    normalize=NORMALIZE)
                distance_matrix[speaker_num * 10 + digit][reference_recording_num] = curr_distance
    return distance_matrix


def get_random_words_true_negatives(threshold: float = THRESHOLD):
    reference_spectrograms = []
    for i in range(10):
        reference_spectrograms.append(get_spectrogram_from_dataset("reference", str(i)))
    true_negatives = 0
    scores = []
    for speaker in SPEAKERS:
        random_spectrogram = get_spectrogram_from_dataset(speaker, "random")
        for reference_spectrogram in reference_spectrograms:
            score = dtw(random_spectrogram, reference_spectrogram, normalize=NORMALIZE)
            # print(speaker, score)
            scores.append(score)
            if score < threshold:
                break
        else:
            true_negatives += 1
    print(min(scores))  # Used for deciding threshold
    return true_negatives


def plot_distance_matrix(distance_matrix, matrix_name: str):
    plt.matshow(distance_matrix, cmap='viridis')
    plt.colorbar(label='Value')
    plt.title(f'Distance matrix of reference to {matrix_name}')
    plt.show()


def create_confusion_matrix(distance_matrix, matrix_name):
    predictions = []
    actuals = []
    for i in range(distance_matrix.shape[0]):
        actuals.append(i % 10)
        min_digit = distance_matrix[i].argmin()
        if min_digit < 10 and distance_matrix[i][min_digit] < THRESHOLD:
            predictions.append(min_digit)
        else:
            predictions.append(10)

    confusion_matrix = np.zeros((11, 11))
    for a, p in zip(actuals, predictions):
        confusion_matrix[a][p] += 1

    plt.figure()
    plt.matshow(confusion_matrix, cmap='Blues', fignum=False)
    plt.title(f'Confusion Matrix - {matrix_name}')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(range(11), [str(i) for i in range(10)] + ['R'])
    plt.yticks(range(11), [str(i) for i in range(10)] + ['R'])
    plt.show()


def classify_dataset():
    distance_matrix = compare_to_reference()
    matrix_name = "Training" if "male_1" in SPEAKERS else "Validation"
    plot_distance_matrix(distance_matrix, matrix_name)
    create_confusion_matrix(distance_matrix, matrix_name)

    true_positives = 0
    true_negatives = get_random_words_true_negatives()
    minimum_digits = distance_matrix.argmin(axis=1)
    for i, min_digit in enumerate(minimum_digits):
        if min_digit == i % 10 and distance_matrix[i][min_digit] < THRESHOLD:
            true_positives += 1
    print(f"{true_positives=}, {true_negatives=}")
    return (true_positives + true_negatives) / (len(minimum_digits) + 4)


if __name__ == '__main__':
    # get_random_words_true_negatives(0)
    # SPEAKERS = TRAINING_SPEAKERS
    # print(f"The training accuracy is: ", classify_dataset())

    SPEAKERS = EVAL_SPEAKERS
    print(f"The evaluation accuracy is: ", classify_dataset())
