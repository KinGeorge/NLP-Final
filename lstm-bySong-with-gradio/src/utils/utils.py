import torch


def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def is_minimum(value, indiv_to_rmse):
    if len(indiv_to_rmse) == 0:
        return True
    temp = list(indiv_to_rmse.values())
    return True if value < min(temp) else False