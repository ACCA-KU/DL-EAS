from rdkit import Chem
import torch
import dgl
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from .MolGraphGenerator import MolGraphGenerator
from D4CMPP2.src.utils.featureizer import InvalidAtomError
from D4CMPP2.src.utils.sculptor import SubgroupSplitter


# ============================================================
# Helper: robust OneHotEncoder
# ------------------------------------------------------------
# sklearn version에 따라 sparse_output 또는 sparse를 사용한다.
# ============================================================

def make_onehot_encoder(categories):
    try:
        return OneHotEncoder(
            sparse_output=False,
            categories=categories,
        )
    except TypeError:
        return OneHotEncoder(
            sparse=False,
            categories=categories,
        )


class ISAGraphGenerator(MolGraphGenerator):
    """
    ISA graph generator modified for LKW-style edge features.

    Key modification:
        r2r edge feature is bond-order scalar:
            single   = 1.0
            double   = 2.0
            triple   = 3.0
            aromatic = 1.5

        Therefore:
            g.edges["r2r"].data["f"].shape = (E_r2r, 1)

    This is required for:
        1. compound r2r GAT
        2. solvent r2r GAT

    Important:
        This version is for the D4CMPP2 modular ISAT pipeline.
        The r2r feature stored in "f" is directly the scalar bond-order feature.
    """

    def __init__(self, frag_ref=None, sculptor_index=(6, 2, 0)):
        # ------------------------------------------------------------
        # Basic flags and dimensions must be defined before super().
        # MolGraphGenerator.__init__() calls self.set_feature_dim().
        # Since set_feature_dim is overridden here, get_graph() may be
        # called during parent initialization.
        # ------------------------------------------------------------

        self.verbose = True

        self.r_node_dim = None
        self.i_node_dim = None
        self.d_node_dim = None

        self.r_edge_dim = None
        self.i_edge_dim = None
        self.d_edge_dim = None

        self.node_dim = None
        self.edge_dim = None

        self.sculptor = SubgroupSplitter(
            frag_ref,
            get_index=True,
            split_order=sculptor_index[0],
            combine_rest_order=sculptor_index[1],
            absorb_neighbor_order=sculptor_index[2],
            overlapped_ring_combine=True,
        )

        # Parent init sets atom featureizer and calls set_feature_dim().
        # Our overridden set_feature_dim() is now safe because all required
        # attributes above are already initialized.
        super().__init__()

        # For compatibility with MolDataManager/ISADataManager config.
        self.node_dim = self.r_node_dim
        self.edge_dim = self.r_edge_dim

    # ------------------------------------------------------------
    # LKW-style bond-order scalar edge feature
    # ------------------------------------------------------------

    def bond_order_scalar(self, bond):
        bt = bond.GetBondType()

        bond_order = {
            Chem.rdchem.BondType.SINGLE: 1.0,
            Chem.rdchem.BondType.DOUBLE: 2.0,
            Chem.rdchem.BondType.TRIPLE: 3.0,
            Chem.rdchem.BondType.AROMATIC: 1.5,
        }

        return bond_order.get(bt, 0.0)

    def get_r2r_edge_features(self, mol, num_r2r_edges):
        """
        Generate directed r2r edge features.

        generate_mol_graph(mol) produces bidirectional molecular edges:
            src + dst, dst + src

        Therefore, bond features are duplicated in the same order:
            bond_features + bond_features

        Returns
        -------
        edata : torch.FloatTensor
            shape = (num_r2r_edges, 1)
        """

        bond_feats = []

        for bond in mol.GetBonds():
            bond_feats.append([self.bond_order_scalar(bond)])

        if len(bond_feats) == 0:
            # Single atom or no-bond case.
            # If the graph has a self-loop, give it neutral edge feature 0.0.
            return torch.zeros((num_r2r_edges, 1), dtype=torch.float32)

        edata = torch.tensor(bond_feats, dtype=torch.float32)
        edata = torch.cat([edata, edata], dim=0)

        if edata.shape[0] != num_r2r_edges:
            raise ValueError(
                "r2r edge feature count mismatch: "
                f"edata rows={edata.shape[0]}, graph r2r edges={num_r2r_edges}"
            )

        return edata

    # ------------------------------------------------------------
    # Feature dimension initialization
    # ------------------------------------------------------------

    def set_feature_dim(self):
        try:
            _ = self.get_graph("FC1CCCCC1CCCOCCC")
        except InvalidAtomError as e:
            if self.verbose:
                print("Invalid atom feature: ", e)

    # ------------------------------------------------------------
    # Main graph generation entry
    # ------------------------------------------------------------

    def get_graph(self, smi, **kwargs):
        if smi == "gas":
            smi = "C"

        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            raise Exception(f"Invalid SMILES: failed to generate mol object: {smi}")

        if kwargs.get("explicit_h", False):
            mol = Chem.AddHs(mol)

        g = self.generate_graph(mol, **kwargs)

        # ------------------------------------------------------------
        # Real atom node feature
        # ------------------------------------------------------------

        atom_feature = self.af(mol)
        g.nodes["r_nd"].data["f"] = torch.tensor(atom_feature).float()

        if self.r_node_dim is None:
            self.r_node_dim = atom_feature.shape[1]

        # ------------------------------------------------------------
        # r2r edge feature: LKW bond-order scalar
        # ------------------------------------------------------------

        num_r2r_edges = g.number_of_edges("r2r")
        edata = self.get_r2r_edge_features(mol, num_r2r_edges)

        g.edges["r2r"].data["f"] = edata.float()

        if self.r_edge_dim is None:
            self.r_edge_dim = edata.shape[1]

        # ------------------------------------------------------------
        # Image node feature
        # ------------------------------------------------------------

        g.nodes["i_nd"].data["f"] = torch.zeros(
            (g.number_of_nodes("i_nd"), 1),
            dtype=torch.float32,
        )

        if self.i_node_dim is None:
            self.i_node_dim = 1

        if self.i_edge_dim is None:
            self.i_edge_dim = 0

        # ------------------------------------------------------------
        # Dot node feature
        # ------------------------------------------------------------

        g.nodes["d_nd"].data["f"] = torch.zeros(
            (g.number_of_nodes("d_nd"), 1),
            dtype=torch.float32,
        )

        if self.d_node_dim is None:
            self.d_node_dim = 1

        # d_edge_dim is set in generate_graph() from d2d distance feature.
        if self.d_edge_dim is None:
            self.d_edge_dim = 0

        return g

    # ------------------------------------------------------------
    # i2i graph among image nodes
    # ------------------------------------------------------------

    def generate_sub_graph(self, mol, frags):
        src, dst = [], []

        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()

            for frag in frags:
                if start in frag:
                    if end in frag:
                        src.append(start)
                        dst.append(end)
                        break
                    else:
                        break

                elif end in frag:
                    if start in frag:
                        src.append(start)
                        dst.append(end)
                        break
                    else:
                        break

        return (src + dst, dst + src)

    # ------------------------------------------------------------
    # d2d graph among dot/fragment nodes
    # ------------------------------------------------------------

    def generate_dot_graph(self, mol, frags, max_dist=4):
        src, dst = [], []

        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()

            start_frag = None
            end_frag = None

            for i, frag in enumerate(frags):
                if start in frag:
                    start_frag = i
                if end in frag:
                    end_frag = i

            if start_frag is not None and end_frag is not None:
                if start_frag != end_frag:
                    src.append(start_frag)
                    dst.append(end_frag)

        if len(src) == 0:
            return ([], []), np.zeros((0, max_dist), dtype=np.float32)

        tmp_g = dgl.graph(
            (src + dst, dst + src),
            num_nodes=len(frags),
        )

        dist = dgl.shortest_dist(tmp_g, root=None)

        src2, dst2, sd_dist = [], [], []

        for i in range(len(dist)):
            for j in range(i + 1, len(dist)):
                if dist[i][j] <= max_dist and dist[i][j] > 0:
                    src2.append(i)
                    dst2.append(j)
                    sd_dist.append(int(dist[i][j]))

        if len(sd_dist) == 0:
            return ([], []), np.zeros((0, max_dist), dtype=np.float32)

        ohe = make_onehot_encoder(
            categories=[list(range(1, max_dist + 1))]
        )

        sd_dist = ohe.fit_transform(
            np.array(sd_dist + sd_dist).reshape(-1, 1)
        ).astype(np.float32)

        return (src2 + dst2, dst2 + src2), sd_dist

    # ------------------------------------------------------------
    # Build ISA heterograph
    # ------------------------------------------------------------

    def generate_graph(self, mol, **kwargs):
        max_dist = kwargs.get("max_dist", 4)
        num_atoms = mol.GetNumAtoms()

        mol_data = self.generate_mol_graph(mol)

        frag = self.sculptor.fragmentation_with_condition(mol)

        if frag is None or len(frag) == 0:
            frag = [[i] for i in range(num_atoms)]

        frag_data = self.generate_sub_graph(mol, frag)

        i2d_src = []
        i2d_dst = []

        for i, f in enumerate(frag):
            for a in f:
                i2d_src.append(int(a))
                i2d_dst.append(i)

        if len(frag) == 1:
            dot_data = ([0], [0])
            dist = np.zeros((1, max_dist), dtype=np.float32)
        else:
            dot_data, dist = self.generate_dot_graph(
                mol,
                frag,
                max_dist=max_dist,
            )

            # If no inter-fragment dot edges were generated, keep a minimal
            # self-loop to avoid empty d2d edge feature problems.
            if len(dot_data[0]) == 0:
                dot_data = ([0], [0])
                dist = np.zeros((1, max_dist), dtype=np.float32)

        graph_data = {
            ("r_nd", "r2r", "r_nd"): mol_data,
            ("r_nd", "r2i", "i_nd"): (
                list(range(num_atoms)),
                list(range(num_atoms)),
            ),
            ("i_nd", "i2i", "i_nd"): frag_data,
            ("i_nd", "i2d", "d_nd"): (i2d_src, i2d_dst),
            ("d_nd", "d2d", "d_nd"): dot_data,
            ("d_nd", "d2r", "r_nd"): (i2d_dst, i2d_src),
        }

        # Explicit node counts make heterograph robust even when some edge
        # types have no edges.
        num_nodes_dict = {
            "r_nd": num_atoms,
            "i_nd": num_atoms,
            "d_nd": len(frag),
        }

        g = dgl.heterograph(
            graph_data,
            num_nodes_dict=num_nodes_dict,
        )

        dist = np.asarray(dist, dtype=np.float32)

        if dist.ndim == 1:
            dist = dist.reshape(-1, max_dist)

        if dist.shape[0] != g.num_edges("d2d"):
            raise ValueError(
                "d2d edge feature count mismatch: "
                f"dist rows={dist.shape[0]}, graph d2d edges={g.num_edges('d2d')}"
            )

        g.edges["d2d"].data["f"] = torch.tensor(dist).float()

        if self.d_edge_dim is None:
            self.d_edge_dim = dist.shape[1] if dist.shape[0] > 0 else max_dist

        return g