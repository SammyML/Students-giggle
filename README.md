# Student Performance Predictor

A machine learning model that predicts students' overall academic performance using demographic and academic data such as gender, parental education level, lunch type, test preparation status, and subject scores.

This project uses a **deep neural network (DNN)** built with TensorFlow and scikit-learn for preprocessing. It trains on historical student data and makes predictions on unseen students.

---

## Features

- Predicts a student's average score based on input data
- Uses a deep learning model (DNN) with dropout regularization
- Explains predictions with plain-language justifications
- Includes preprocessing pipeline for both categorical and numerical data

---

## Model Architecture

- Input layer (preprocessed features)
- Dense layer (128 neurons, ReLU)
- Dropout layer (0.2)
- Dense layer (64 neurons, ReLU)
- Dropout layer (0.2)
- Dense layer (32 neurons, ReLU)
- Output layer (1 neuron, for regression)

---

## Dataset

The dataset used is a synthetic version of the popular [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance), including fields like:

- Gender
- Race/Ethnicity
- Parental level of education
- Lunch (standard / free-reduced)
- Test preparation course
- Math, Reading, and Writing scores

---

## How to Run

### Requirements

Make sure you have Python 3.10 or 3.11 installed.

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow pandas scikit-learn
