import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from rdkit import Chem
from smiles_utils import _convertToAdj, _convertToFeatures
from model_def import GCNNet_Eval


def smiles_to_XA(smiles_str: str) -> torch.Tensor:
    adj = _convertToAdj([smiles_str])
    feats, _ = _convertToFeatures([smiles_str])
    X_A = np.concatenate([feats.astype(np.float32), adj.astype(np.float32)], axis=1)
    return torch.from_numpy(X_A).unsqueeze(0).contiguous()


class MolGraphDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        s = self.data.iloc[idx]
        return s['Chromophore_smiles'], s['Solvent_smiles'], np.array(s[2:], dtype=np.float32)


file_path = "data/spectra.csv"
dataset = MolGraphDataset(file_path)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

ckpt_path = "model/pretrained_weight.ckpt"
model = GCNNet_Eval.load_from_checkpoint(ckpt_path)
model.eval().cuda()


def norm_mae(y, pred):
    y, pred = torch.tensor(y), torch.tensor(pred)
    y_max = torch.max(torch.nan_to_num(y))
    mask = ~torch.isnan(y) & ~torch.isnan(pred)
    return torch.mean(torch.abs((y[mask]/y_max)-(pred[mask]/y_max))).item()


predictions, maes, ys = [], [], []
for smi, sol, y in loader:
    X_A, X_A_sol = smiles_to_XA(smi[0]).cuda(), smiles_to_XA(sol[0]).cuda()
    with torch.no_grad():
        y_hat = model(X_A, X_A_sol).cpu().numpy()
    predictions.append(y_hat)
    ys.append(y)
    maes.append(norm_mae(y, y_hat))

print(f"Average Norm-MAE: {np.mean(maes):.4f}")


x = np.linspace(330, 1000, 336)
i = 0 # any test index
plt.scatter(x, ys[i][0], s=1, label='True')
plt.scatter(x, predictions[i][0], s=1, label='Predicted')
plt.legend()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Extinction coefficient (L·mol⁻¹·cm⁻¹)")
plt.title(f"DL-EAS Prediction Example (Index = {i})")
plt.show()
