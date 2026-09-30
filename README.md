# Activity Anomaly Detection Baseline

This repository contains a synthetic activity dataset and a notebook that trains and evaluates a supervised LSTM baseline for classifying activity sessions as normal or anomalous.

## Repository Contents

- `01_anomaly_baseline.ipynb`: data inspection, preprocessing, model training, evaluation, and an example prediction.
- `activity_sessions.csv`: 5,000 session-level records with event sequences, session metadata, and an `is_anomaly` label.
- `activity_events.csv`: 32,663 event-level records.
- `event_vocab.csv`: event-to-integer vocabulary supplied with the dataset.

## Dataset

The notebook uses `activity_sessions.csv`. It contains 4,418 normal sessions (88.36%) and 582 anomalous sessions (11.64%). The saved notebook output reports no missing values in the session dataset. Sequence lengths range from 5 to 8 events, with a mean length of 6.53.

## Baseline Workflow

The notebook performs these steps:

1. Reads the session CSV and inspects labels, sequence lengths, and example sequences.
2. Builds an event-to-integer mapping from the observed event names and encodes each sequence. The provided `event_vocab.csv` is included in the dataset but is not loaded by this notebook.
3. Pads or truncates sequences to 10 event IDs, using 0 for padding.
4. Creates stratified train, validation, and test splits with a fixed random seed: 70% / 15% / 15% (3,500 / 750 / 750 sessions).
5. Trains an embedding layer (32 dimensions), an LSTM (64 units), and a sigmoid output layer using Adam and binary cross-entropy for 20 epochs.
6. Uses the sigmoid output as an anomaly score and classifies scores above 0.5 as anomalous. It reports classification metrics and plots score distributions and a confusion matrix.

This is a **supervised** classifier: the training labels are `is_anomaly`. It is not the normal-only sequence prediction experiment described in some anomaly-detection approaches.

## Saved Notebook Results

On the notebook's stratified test split, the saved output reports:

| Metric | Result |
| --- | ---: |
| Test sessions | 750 |
| Test loss | 0.00000996 |
| Test accuracy | 100% |
| Normal precision / recall / F1 | 1.00 / 1.00 / 1.00 (663 sessions) |
| Anomaly precision / recall / F1 | 1.00 / 1.00 / 1.00 (87 sessions) |

The notebook's example sequence (`LOGIN PULL READ_FILE COMMIT`) received an anomaly probability of approximately `0.000004` and was classified as normal.

These are results from the saved run, not a guarantee that a fresh run or real activity data will achieve the same performance. Because the dataset is synthetic and the test split is a random split from that same dataset, the perfect score may reflect patterns specific to how the data was generated. Validate on independently generated or real-world data before relying on this model.

## Run the Notebook

Use Python with Jupyter and the notebook's dependencies installed: TensorFlow, NumPy, pandas, Matplotlib, Seaborn, and scikit-learn. From this directory, start Jupyter and open the notebook:

```bash
jupyter lab 01_anomaly_baseline.ipynb
```

Run the cells from top to bottom. The notebook expects `activity_sessions.csv` to be available in the current working directory. Training is configured to use the CPU.