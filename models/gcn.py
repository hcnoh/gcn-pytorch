import torch

from torch.nn import Module, Linear
from torch.sparse import mm


class GCN(Module):
    '''
        Args:
            A_hat: A preprocessed adjacency matrix with self-connections and
            renormalization trick.
            C: The number of the input channel.
            H: The number of the feature maps of hidden layers
            F: The number of the feature maps of output layer.
    '''
    def __init__(self, A_hat, C, H, F):
        super().__init__()

        self.A_hat = torch.tensor(A_hat).to_sparse()
        self.C = C
        self.H = H
        self.F = F

        self.W0 = Linear(self.C, self.H, bias=False)
        self.W1 = Linear(self.H, self.F, bias=False)

    def forward(self, X):
        Z = torch.softmax(
            mm(
                self.A_hat,
                self.W1(torch.relu(mm(self.A_hat, self.W0(X))))
            )
        )

        return Z
