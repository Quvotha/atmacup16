import numpy as np


def ap10(grand_truth: np.ndarray[np.int64], predictions: np.ndarray[np.int64]) -> np.ndarray[np.float64]:
    """Calculate AP@10 for each row.


    Parameters
    ----------
    grand_truth : np.ndarray[np.int64], N * 1 matrix.
        Actual label.
    predictions : np.ndarray[np.int64], N * K matrix.
        Your prediction matrix, order dose matter.


    Returns
    -------
    ap_array : np.ndarray[np.float64], N * 1 matrix.
        AP@K of each row.
    """
    assert len(grand_truth) == len(predictions)
    k = predictions.shape[1]
    grand_truth_reshape = np.tile(grand_truth[:, np.newaxis], (1, k))
    comparison = grand_truth_reshape == predictions
    mask = np.any(comparison, axis=1)  # True: 予測に正解が含まれている
    apk = mask * (1 / (1 + np.argmax(comparison, axis=1)))
    return apk.astype(np.float64)
