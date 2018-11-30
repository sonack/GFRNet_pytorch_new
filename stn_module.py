import torch

class STN(torch.nn.Module):
    def __init__(self):
        super(STN, self).__init__()

    def forward(self, input, grid):
        return torch.nn.functional.grid_sample(input, grid)