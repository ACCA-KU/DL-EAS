import torch
from torch import nn
import pytorch_lightning as pl

class GCN(nn.Module):
    def __init__(self, unit, input_shape, activation=nn.ReLU(), is_bias=False):
        super(GCN, self).__init__()
        self.hidden_dim = unit
        self.input_shape = input_shape
        self.activation = activation
        self.Linear = nn.Linear(input_shape, self.hidden_dim, bias=is_bias)
        if unit != input_shape:
            self.Linear_dim = nn.Linear(input_shape, self.hidden_dim, bias=is_bias)
            nn.init.xavier_uniform_(self.Linear_dim.weight)
        nn.init.xavier_uniform_(self.Linear.weight)

    def forward(self, X, A):
        mat = torch.matmul(A, self.Linear(X))
        out = self.activation(mat)
        out += X if self.hidden_dim == self.input_shape else self.Linear_dim(X)
        return out


class GCNNet_Eval(pl.LightningModule):
    def __init__(self, pred_dim=336):
        super().__init__()
        self.pred_dim = pred_dim
        self.depth = 6
        self.dims = [47] + [128] * self.depth
        act = nn.LeakyReLU(0.1)

        self.GCN_list = nn.ModuleList(
            [GCN(self.dims[i + 1], self.dims[i], activation=act) for i in range(len(self.dims) - 1)]
        )
        self.sol_GCN_list = nn.ModuleList(
            [GCN(self.dims[i + 1], self.dims[i], activation=act) for i in range(len(self.dims) - 1)]
        )

        self.linear_list = nn.Sequential(
            nn.Linear(self.dims[-1], 1024), act, nn.BatchNorm1d(1024),
            nn.Linear(1024, 512), act, nn.BatchNorm1d(512)
        )
        self.sol_linear_list = nn.Sequential(
            nn.Linear(self.dims[-1], 1024), act, nn.BatchNorm1d(1024),
            nn.Linear(1024, 512), act, nn.BatchNorm1d(512)
        )
        self.pred_list = nn.Sequential(
            nn.Linear(1024, 512), act, nn.BatchNorm1d(512),
            nn.Linear(512, 1024), act, nn.Dropout(0.5),
            nn.Linear(1024, 2048), act, nn.Dropout(0.4),
            nn.Linear(2048, 1024), act, nn.Dropout(0.2),
            nn.Linear(1024, pred_dim)
        )

    def forward(self, X_A, X_A_sol):
        X, A = X_A[:, :, :47], X_A[:, :, 47:]
        Xs, As = X_A_sol[:, :, :47], X_A_sol[:, :, 47:]
        H, Hs = X, Xs
        for G, Gs in zip(self.GCN_list, self.sol_GCN_list):
            H, Hs = G(H, A), Gs(Hs, As)
        H, Hs = torch.sum(H, -2), torch.sum(Hs, -2)
        H, Hs = self.linear_list(H), self.sol_linear_list(Hs)
        return self.pred_list(torch.cat([H, Hs], -1))
