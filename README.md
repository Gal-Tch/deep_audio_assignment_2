# Advanced Topics in Audio Processing using Deep Learning - Assignment 2

This project focuses on Automatic Speech Recognition (ASR) techniques, specifically implementing Dynamic Time Warping (DTW) and investigating Connectionist Temporal Classification (CTC) and Force Alignment.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Setup Guide](#setup-guide)
- [Directory Structure](#directory-structure)
- [File Explanations](#file-explanations)
- [Usage](#usage)

## 🎯 Project Overview
The assignment involves:
1. **Data Acquisition**: Recording and processing audio digits (0-9) and random words.
2. **DTW Implementation**: Building a custom DTW algorithm to classify digits based on a reference database.
3. **CTC Forward Algorithm**: Implementing the collapse function and forward pass for sequence probability calculation.
4. **Force Alignment**: Adapting CTC for path finding and alignment on specific datasets.

## 🛠️ Setup Guide

### Prerequisites
- **Python 3.10** is recommended for this assignment.

### Installation
1.  **Clone the repository** (if applicable) or navigate to the project directory.
2.  **Create a virtual environment** (optional but recommended):
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```powershell
    pip install -r requirements.txt
    ```

### Data Preparation
The scripts expect a specific directory structure for audio files:
- Place raw recordings in `raw_recordings/`.
- Run `data_acquisition.py` to process these into the `dataset/` directory (resampling to 16kHz and converting to .wav).

## 📁 Directory Structure
```text
deep_audio_assignment_2/
├── raw_recordings/      # Original audio files (various formats)
├── dataset/             # Processed 16kHz WAV files (mirrors raw_recordings structure)
├── results/             # Generated plots and output files
├── data_acquisition.py  # Script for audio processing and Mel Spectrogram analysis
├── dtw.py               # Implementation of DTW and digit classification
├── ctc.py               # Implementation of CTC Forward Algorithm and Force Alignment
├── force_align.pkl      # Data for the Force Alignment task
├── requirements.txt     # Python dependencies
├── Assignment 2.md     # Detailed assignment instructions
└── README.md            # This file
```

## 📄 File Explanations

- **`data_acquisition.py`**:
  - Handles resampling of audio to 16kHz.
  - Computes Mel Spectrograms (25ms window, 10ms hop, 80 filter banks).
  - Provides visualization functions for comparing spectrograms.
- **`dtw.py`**:
  - Contains the core logic for the Dynamic Time Warping algorithm.
  - Performs classification of the training set against a reference set.
  - Calculates accuracy and handles "random word" rejection via thresholds.
- **`ctc.py`**:
  - Implements the CTC collapse function.
  - Implements the CTC Forward Algorithm for sequence probability.
  - Implements Force Alignment using the `max` operator and backtracking.
  - Processes `force_align.pkl` for real-data demonstration.
- **`requirements.txt`**: List of libraries needed, including `librosa`, `soundfile`, `numba`, and `matplotlib`.
- **`Assignment 2.md`**: The original assignment prompt and requirements.

## 🚀 Usage

### 1. Process Raw Audio
To convert your raw recordings and view spectrogram differences:
```powershell
python data_acquisition.py
```
*(Note: Ensure `create_dataset()` is called in `if __name__ == "__main__":` if you haven't processed your files yet.)*

### 2. Run DTW Classification
To perform digit classification and see the accuracy results:
```powershell
python dtw.py
```

### 3. Run CTC & Force Alignment
To run the CTC forward pass and force alignment (including the real data sample):
```powershell
python ctc.py
```
Results (plots and logs) will be saved in the `results/` directory.

---
**Author:** Tal Rosenwein
**Assignment:** ASR (Technical Part - Python 3.10)
