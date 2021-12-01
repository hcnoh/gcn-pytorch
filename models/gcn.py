import torch

from torch.nn import Module, Linear, Dropout
from torch.sparse import mm
from torch.optim import Adam

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor, LongTensor


class GCN(Module):
    '''
        Args:
            A_hat: A preprocessed adjacency matrix with self-connections and
            renormalization trick.
            C: The number of the input channel(input feature maps).
            H: The number of the feature maps of hidden layers
            F: The number of the feature maps of output layer.
    '''
    def __init__(self, A_hat, C, H, F, num_layers, dropout, regularization):
        super().__init__()

        self.A_hat = FloatTensor(A_hat).to_sparse()
        self.C = C
        self.H = H
        self.F = F

        self.num_layers = num_layers
        self.dropout = dropout
        self.regularization = regularization

        self.W0 = Linear(self.C, self.H, bias=False)
        self.W1 = Linear(self.H, self.F, bias=False)

        self.Wh = [
            Linear(self.H, self.H, bias=False)
            for _ in range(self.num_layers - 2)
        ]

        self.dropout_layer = Dropout(self.dropout)

        self.L2 = torch.sum(
            FloatTensor([torch.norm(param) for param in self.W0.parameters()])
        )

    def forward(self, X):
        Z = self.dropout_layer(torch.relu(mm(self.A_hat, self.W0(X))))
        for Wh in self.Wh:
            Z = torch.relu(mm(self.A_hat, Wh(Z)))
        Z = self.dropout_layer(
            torch.softmax(mm(self.A_hat, self.W1(Z)), dim=-1)
        )

        return Z

    def train_model(
        self, num_epochs, learning_rate, dataset, train_indices, test_indices
    ):
        opt = Adam(self.parameters(), learning_rate)

        X = FloatTensor(dataset.X)

        for i in range(1, num_epochs + 1):
            self.eval()

            _, Y = dataset[train_indices]
            Y = FloatTensor(Y)

            Z = torch.gather(
                self(X), dim=0, index=LongTensor(train_indices).unsqueeze(-1).repeat(1, self.F)
            )

            print(self(X).shape, Z.shape, train_indices.shape, self(X)[train_indices[0]] == Z[0])
