# Activity Anomaly Dataset

Synthetic dataset generated for the Git/Server Activity Anomaly Detector.

Files:
- activity_events.csv: 32,663 event-level records
- activity_sessions.csv: 5,000 session-level sequences
- event_vocab.csv: event-to-integer vocabulary

Dataset statistics:
- Sessions: 5,000
- Events: 32,663
- Anomalous sessions: 582 (11.64%)
- Normal sessions: 4,418

Suggested first experiment:
1. Use activity_sessions.csv.
2. Train primarily on normal sequences.
3. Convert event names to IDs using event_vocab.csv.
4. Build fixed-length sequences.
5. Train an LSTM and GRU.
6. Use prediction error/probability as an anomaly score.
