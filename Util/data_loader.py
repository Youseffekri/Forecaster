"""
Data Loader for the Neural Models

Builds the train/test tensor dictionary the torch-based models (`MHAttn_Regressor`) consume.
torch is imported lazily, so the rest of the package does not need it (`pip install -e .[neural]`).

Author: Yousef Fekri Dabanloo
Date: 2025-03-05 (rev 2026-10-12)
Version: 1.1
License: MIT License
"""

from sklearn.model_selection import train_test_split


def Data_Loader(X, Y, TnT=False, shuffle=False, device="cpu"):
    """
    Converts (X, Y) into a dict of torch tensors: train_input, test_input, train_label, test_label
    (an 80/20 split when TnT, else train = test = the full data).

    Parameters
    ----------
    X, Y : np.ndarray
        The inputs and the labels.
    TnT : bool, default=False
        Whether to split into training and testing sets (80 % / 20 %).
    shuffle : bool, default=False
        Whether the split shuffles the rows (keep False for time series).
    device : str, default="cpu"
        The torch device.
    """
    import torch                                                   # optional dependency

    if TnT:
        train_input, test_input, train_label, test_label = train_test_split(X, Y, test_size=0.2, shuffle=shuffle)
    else:
        train_input, test_input, train_label, test_label = X, X, Y, Y

    dtype = torch.get_default_dtype()
    return {"train_input": torch.from_numpy(train_input).type(dtype).to(device),
            "test_input": torch.from_numpy(test_input).type(dtype).to(device),
            "train_label": torch.from_numpy(train_label).type(dtype).to(device),
            "test_label": torch.from_numpy(test_label).type(dtype).to(device)}
