import torch
import torch.nn as nn
import random


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    @torch.no_grad()
    def predict_test(self) -> float:
        return float(random.random())
