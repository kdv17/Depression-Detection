# Tri-Modal Time-Series Depression Severity Prediction

## 📌 Project Overview
This repository contains the code and methodology for an automated, multimodal machine learning pipeline designed to estimate depression severity. By tracking how a patient's language, voice, and facial expressions change over the course of a clinical interview, the system predicts their **PHQ-8 score** (a standard clinical measure of depression severity). 

This project transitions from classical machine learning baselines (Gradient Boosting, Random Forest) to a robust **Deep Learning Time-Series Framework** (BiLSTMs and CNNs) to capture longitudinal psychological cues.

## 📊 The Dataset: E-DAIC
This project utilizes the **Extended Distress Analysis Interview Corpus (E-DAIC)**. The data consists of clinical interviews conducted by an animated virtual agent ("Ellie").
### Extracted Features Used:
* **Text:** Timestamped patient utterances and clinical summaries.
* **Audio:** Mel-Frequency Cepstral Coefficients (MFCCs) and Bag-of-Audio-Words (BoAW) extracted via **OpenSMILE 2.3.0** (sampled at 0.1s intervals).
* **Video:** Facial Action Units (AUs), Head Pose, and Eye Gaze extracted frame-by-frame via **OpenFace 2.1.0** (filtered for confidence > 0.8).

## 🧠 Model Architecture
Our framework processes the three modalities independently before fusing them for a final prediction.

1. **Text Modality (The Words):** * Tracks the sequence of spoken utterances.
   * **Model:** BiLSTM (Bidirectional Long Short-Term Memory) to capture contextual semantic shifts over the interview timeline.
2. **Audio Modality (The Voice):**
   * Converts acoustic data into log-mel spectrograms (time-frequency visual maps).
   * **Model:** 2D CNN (Convolutional Neural Network) to detect prosodic markers such as monotonic speech or reduced vocal energy.
3. **Video Modality (The Face):**
   * Tracks the timeline of physical expressivity (Action Units, averted gaze).
   * **Model:** BiLSTM to capture temporal facial dynamics and psychomotor retardation.
4. **Tri-Modal Late Fusion:**
   * The independent outputs of the Text, Audio, and Video models are concatenated and passed through a Multi-Layer Perceptron (MLP) to output the final continuous PHQ-8 score.

