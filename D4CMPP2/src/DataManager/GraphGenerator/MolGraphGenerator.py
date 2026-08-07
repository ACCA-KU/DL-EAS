import dgl
import torch
import rdkit.Chem as Chem
import numpy as np
import traceback

from D4CMPP2.src.utils.featureizer import get_atom_features, InvalidAtomError


class MolGraphGenerator:
    def __init__(self):
        self.af = get_atom_features
        self.set_feature_dim()

    # ------------------------------------------------------------
    # LKW-style bond-order scalar edge feature
    # ------------------------------------------------------------
    def bond_features(self, bond):
        bt = bond.GetBondType()

        bond_order = {
            Chem.rdchem.BondType.SINGLE: 1.0,
            Chem.rdchem.BondType.DOUBLE: 2.0,
            Chem.rdchem.BondType.TRIPLE: 3.0,
            Chem.rdchem.BondType.AROMATIC: 1.5,
        }

        return bond_order.get(bt, 0.0)

    # Set the feature dimensions by generating a dummy graph
    def set_feature_dim(self):
        self.node_dim = self.af(Chem.MolFromSmiles('C')).shape[1]

        # LKW-style edge feature:
        # one scalar bond-order feature per directed edge.
        self.edge_dim = 1

    # Get the graph from the SMILES
    def get_graph(self, smi, **kwargs):
        if smi == "gas":
            smi = "C"

        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            raise Exception("Invalid SMILES: failed to generate mol object")

        if kwargs.get("explicit_h", False) or mol.GetNumAtoms() == 1:
            mol = Chem.AddHs(mol)

        g = self.generate_graph(mol)
        g = self.add_feature(g, mol)

        return g

    def get_empty_graph(self):
        g = dgl.graph(([], []))
        g.ndata['f'] = torch.zeros((0, self.node_dim)).float()
        g.edata['f'] = torch.zeros((0, self.edge_dim)).float()
        return g

    # Add the features to the graph
    def add_feature(self, g, mol):
        atom_feature = self.af(mol)
        g.ndata['f'] = torch.tensor(atom_feature).float()

        bond_x = []

        if mol.GetNumBonds() == 0:
            # In generate_mol_graph(), single-atom molecules receive a self-loop.
            # Give that self-loop a neutral edge feature 0.0.
            if g.num_edges() > 0:
                bond_x = [0.0 for _ in range(g.num_edges())]
                g.edata['f'] = torch.tensor(bond_x, dtype=torch.float32).view(-1, 1)
            else:
                g.edata['f'] = torch.zeros((0, self.edge_dim)).float()
        else:
            for bond in mol.GetBonds():
                feat = self.bond_features(bond)

                # generate_mol_graph() creates bidirectional edges:
                # src+dst, dst+src
                # Therefore edge features must also be duplicated.
                bond_x.append(feat)

            edata = torch.tensor(bond_x, dtype=torch.float32).view(-1, 1)
            edata = torch.cat([edata, edata], dim=0)
            g.edata['f'] = edata

        return g

    # Generate the graph from the molecule object
    def generate_mol_graph(self, mol):
        num_atoms = mol.GetNumAtoms()

        if num_atoms == 1:
            return ([0], [0])

        src, dst = [], []

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src.append(start)
            dst.append(end)

        return (src + dst, dst + src)

    def generate_graph(self, mol):
        mol_data = self.generate_mol_graph(mol)
        g = dgl.graph(mol_data, num_nodes=mol.GetNumAtoms())
        return g