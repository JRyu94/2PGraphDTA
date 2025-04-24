import numpy as np

def concordance_index(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    num_pairs, num_correct = 0, 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] != y_true[j]:
                num_pairs += 1
                if (y_pred[i] - y_pred[j]) * (y_true[i] - y_true[j]) > 0:
                    num_correct += 1
                elif (y_pred[i] - y_pred[j]) == 0:
                    num_correct += 0.5
    return num_correct / num_pairs if num_pairs > 0 else 0.0

def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

def mean_std(arr):
    return np.mean(arr), np.std(arr)