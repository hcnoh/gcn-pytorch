import os

import numpy as np
import torch

from torch.nn import Module, Linear, Dropout
from torch.nn.init import xavier_normal_
from torch.nn.functional import cross_entropy
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

        xavier_normal_(self.W0.weight)
        xavier_normal_(self.W1.weight)

        self.Wh = [
            Linear(self.H, self.H, bias=False)
            for _ in range(self.num_layers - 2)
        ]

        for Wh in self.Wh:
            xavier_normal_(Wh.weight)

        self.dropout_layer = Dropout(self.dropout)

        self.L2 = torch.sum(
            FloatTensor([torch.norm(param) for param in self.W0.parameters()])
        )

    def get_logits(self, X):
        Z = self.dropout_layer(torch.relu(mm(self.A_hat, self.W0(X))))
        for Wh in self.Wh:
            Z = torch.relu(mm(self.A_hat, Wh(Z)))
        Z = self.dropout_layer(mm(self.A_hat, self.W1(Z)))

        return Z

    def forward(self, X):
        Z = self.get_logits(X)

        return torch.softmax(Z, dim=-1)

    def train_model(
        self, num_epochs, learning_rate, dataset, train_indices, test_indices,
        ckpt_path
    ):
        accs = []
        train_losses = []
        test_losses = []

        max_acc = 0

        opt = Adam(self.parameters(), learning_rate)

        X = FloatTensor(dataset.X)

        for i in range(1, num_epochs + 1):
            self.train()

            _, Y = dataset[train_indices]
            Y = LongTensor(Y)

            Z = torch.gather(
                self.get_logits(X),
                dim=0,
                index=LongTensor(train_indices)
                .unsqueeze(-1).repeat(1, self.F)
            )

            opt.zero_grad()
            train_loss = cross_entropy(Z, Y)
            (train_loss + self.regularization * self.L2).backward()
            opt.step()

            train_loss = train_loss.detach().cpu().numpy()

            train_losses.append(train_loss)

            with torch.no_grad():
                self.eval()

                _, Y = dataset[test_indices]
                Y = LongTensor(Y)

                Z = torch.gather(
                    self.get_logits(X),
                    dim=0,
                    index=LongTensor(test_indices)
                    .unsqueeze(-1).repeat(1, self.F)
                )

                test_loss = cross_entropy(Z, Y)
                test_loss = test_loss.detach().cpu().numpy()

                test_losses.append(test_loss)

                Y = Y.detach().cpu().numpy()

                Z = torch.softmax(Z, dim=-1).detach().cpu().numpy()
                Z = np.argmax(Z, axis=-1)

                acc = np.mean(Y == Z)

                accs.append(acc)

                print(
                    "Epoch: {}, Train Loss: {}, Test Loss: {}, Test ACC: {}"
                    .format(i, train_loss, test_loss, acc)
                )

                if acc > max_acc:
                    torch.save(
                        self.state_dict(),
                        os.path.join(
                            ckpt_path, "model.ckpt"
                        )
                    )
                    max_acc = acc

        return accs, train_losses, test_losses
