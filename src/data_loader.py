import numpy as np
import torch

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of data from the dataset."""
    if dataset.ndim != 1:
        raise ValueError("Dataset must be a 1D array.")
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    if context_length <= 0:
        raise ValueError("Context length must be a positive integer.")
    if len(dataset) < context_length + 1:
        raise ValueError("Dataset must be at least as large as context length + 1.")

    start_indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    inputs = []
    targets = []
    for start in start_indices:
        inputs.append(dataset[start : start + context_length])
        targets.append(dataset[start + 1 : start + context_length + 1])
    # stack作用：将数组沿着新的轴连接起来，形成一个新的数组。这里将inputs和targets分别堆叠成一个二维数组，每一行对应一个样本。
    inputs = torch.from_numpy(np.stack(inputs)).to(device=device, dtype=torch.long)
    targets = torch.from_numpy(np.stack(targets)).to(device=device, dtype=torch.long)

    return inputs, targets