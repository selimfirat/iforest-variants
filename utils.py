import numpy as np

def label_first_N_anomaly(anomaly_scores, N): # N anomalies

    mask = np.argsort(-anomaly_scores)[:N]

    y = np.zeros(anomaly_scores.shape[0])

    y[mask] = 1

    return y