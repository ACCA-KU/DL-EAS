import math
import torch
import torch.nn as nn

import dgl
from dgl.nn import SumPooling

from D4CMPP2.networks.src.GCN import GCN_layer
from D4CMPP2.networks.src.distGCN import distGCN_layer
from D4CMPP2.networks.src.GAT import GATs

try:
    from D4CMPP2.networks.src.Linear import Linears
except Exception:
    from networks.src.Linear import Linears


def get_activation(name="leakyrelu", negative_slope=0.1):
    """
    Configurable MLP activation.

    Recommended R3 PM6-free setting:
        mlp_activation = "leakyrelu"

    Supported:
        "leakyrelu", "relu", "gelu", "silu"
    """
    name = str(name).lower()

    if name in ["leakyrelu", "leaky_relu", "lrelu"]:
        return nn.LeakyReLU(negative_slope)
    elif name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name in ["silu", "swish"]:
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation: {name}")


class network(nn.Module):
    """
    DLEAS_ISAT_R3_PM6FREE_SplitAblation_model

    Purpose:
        PM6-free split-ablation version of DLEAS_ISAT_R3_PM6FREE_model.

    Main design:
        1. Compound r2r atom-level message passing:
           edge-aware additive GAT.

        2. Compound i2i:
           original GCN retained.
           R3 default:
               i2i_residual=True

        3. d2score:
           vector gate.
           R3 default:
               d2r context -> Linears(hidden_dim -> hidden_dim)
               -> BatchNorm1d(hidden_dim)
               -> sigmoid
               -> r_node * score

        4. Solvent branch:
           6-layer edge-aware GAT.

        5. PM6-free spectral query:
           spectra R3:
               spectrum_att = mol_gate * calc_spec

           PM6-free R3:
               spectrum_att = mol_feat

           Therefore, c330_var ... c1000_var may exist in the batch,
           but they are intentionally ignored by this model.

        6. Split final linear:
           R3 default:
               use_split_final_linear=False

        7. Final correction:
           R3_PM6FREE:
               wavelength-local Conv1D correction head

        8. Split ablation:
           Default:
               spec_split_dims = (86, 100, 150)

           Optional:
               spec_split_dims can be passed from DLEAS_semi.train(...)

           Examples:
               spec_split_dims=(85, 135, 116)
               spec_split_dims=(61, 100, 175)
               spec_split_dims=(48, 86, 202)

        9. Activation/dropout:
           R3 default:
               mlp_activation="leakyrelu"
               dropout=0.1
               gat_dropout=0.0

        10. Loss:
            masked MAE.

        11. Spectrum dimension:
            330–1000 nm, 2 nm interval = 336 points.
    """

    def __init__(self, config):
        super(network, self).__init__()

        hidden_dim = config.get("hidden_dim", 128)
        gcn_layers = config.get("conv_layers", 6)

        # R3 default aligned with the selected R3 spectra model.
        dropout = config.get("dropout", 0.1)
        gat_dropout = config.get("gat_dropout", 0.0)
        linear_layers = config.get("linear_layers", 3)

        edge_dim = config.get("edge_dim", 1)
        d_edge_dim = config.get("d_edge_dim", 4)
        heads = config.get("heads", config.get("num_heads", 8))
        target_dim = config["target_dim"]

        # ------------------------------------------------------------------
        # R3 architecture switches
        # ------------------------------------------------------------------
        self.i2i_residual = config.get("i2i_residual", True)
        self.d2score_mode = config.get("d2score_mode", "vector_gate")
        self.use_split_final_linear = config.get("use_split_final_linear", False)
        self.mlp_activation_name = config.get("mlp_activation", "leakyrelu")
        self.gat_node_activation_slope = config.get("gat_node_activation_slope", 0.1)

        if self.d2score_mode not in ["vector_gate", "scalar_gate"]:
            raise ValueError(
                f"d2score_mode must be 'vector_gate' or 'scalar_gate', "
                f"got {self.d2score_mode}"
            )

        # ------------------------------------------------------------------
        # PM6-free flags
        # ------------------------------------------------------------------
        self.use_pm6_spectrum = False
        self.query_source = "graph_mol_feat"
        self.used_pm6_spectrum = False
        self.ignored_pm6_key_count = 0

        # ------------------------------------------------------------------
        # R3-specific Conv1D correction settings
        # ------------------------------------------------------------------
        self.spectral_correction_mode = config.get(
            "spectral_correction_mode",
            "conv1d_local",
        )

        self.conv_correction_channels = int(config.get("conv_correction_channels", 16))
        self.conv_kernel_size = int(config.get("conv_kernel_size", 7))
        self.use_learnable_conv_scale = bool(config.get("use_learnable_conv_scale", True))
        self.conv_correction_scale_init = float(config.get("conv_correction_scale_init", 0.1))

        if self.conv_kernel_size % 2 == 0:
            raise ValueError(
                f"conv_kernel_size should be odd for same-length padding, "
                f"got {self.conv_kernel_size}"
            )

        self.conv_padding = self.conv_kernel_size // 2

        if self.use_learnable_conv_scale:
            self.conv_correction_scale = nn.Parameter(
                torch.tensor(self.conv_correction_scale_init, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "conv_correction_scale",
                torch.tensor(self.conv_correction_scale_init, dtype=torch.float32),
            )

        self.alpha = config.get("alpha", 1e-2)
        self.target_dim = target_dim
        self.spec_dim = target_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.d_edge_dim = d_edge_dim
        self.heads = heads
        self.dropout = dropout
        self.gat_dropout = gat_dropout
        self.linear_layers = linear_layers

        # ------------------------------------------------------------------
        # Spectrum split setting
        # ------------------------------------------------------------------
        # Default R3 split:
        #   336-point spectrum: 86 + 100 + 150 = 336.
        #
        # Split-ablation mode:
        #   pass spec_split_dims from DLEAS_semi.train(...)
        #
        # Examples:
        #   spec_split_dims=(85, 135, 116)
        #   spec_split_dims=(61, 100, 175)
        #   spec_split_dims=(48, 86, 202)
        # ------------------------------------------------------------------
        spec_split_dims = config.get("spec_split_dims", None)

        if spec_split_dims is None:
            if target_dim == 336:
                self.spec_split_dims = (86, 100, 150)
            else:
                a = target_dim // 3
                b = target_dim // 3
                c = target_dim - a - b
                self.spec_split_dims = (a, b, c)
        else:
            self.spec_split_dims = tuple(int(x) for x in spec_split_dims)

        if len(self.spec_split_dims) != 3:
            raise ValueError(
                f"spec_split_dims must have length 3, got {self.spec_split_dims}"
            )

        if sum(self.spec_split_dims) != target_dim:
            raise ValueError(
                f"sum(spec_split_dims) must match target_dim. "
                f"got spec_split_dims={self.spec_split_dims}, "
                f"sum={sum(self.spec_split_dims)}, target_dim={target_dim}"
            )

        spec_dim1, spec_dim2, spec_dim3 = self.spec_split_dims

        mlp_act = get_activation(self.mlp_activation_name, self.gat_node_activation_slope)

        print("=" * 100)
        print("[DLEAS_ISAT_R3_PM6FREE_SplitAblation_model CONFIG CHECK - LKW ISAT1 PM6-FREE solGAT/r2rGAT R3]")
        print(f"node_dim                  = {config['node_dim']}")
        print(f"edge_dim                  = {edge_dim}")
        print(f"d_edge_dim                = {d_edge_dim}")
        print(f"target_dim                = {target_dim}")
        print(f"hidden_dim                = {hidden_dim}")
        print(f"conv_layers               = {gcn_layers}")
        print(f"heads                     = {heads}")
        print(f"dropout                   = {dropout}")
        print(f"gat_dropout               = {gat_dropout}")
        print(f"linear_layers             = {linear_layers}")
        print(f"spec_dim                  = {self.spec_dim}")
        print(f"spec_split_dims           = {self.spec_split_dims}")
        print(f"i2i_residual              = {self.i2i_residual}")
        print(f"d2score_mode              = {self.d2score_mode}")
        print(f"use_split_final_linear    = {self.use_split_final_linear}")
        print(f"mlp_activation            = {self.mlp_activation_name}")
        print(f"use_pm6_spectrum          = {self.use_pm6_spectrum}")
        print(f"query_source              = {self.query_source}")
        print(f"spectral_correction_mode  = {self.spectral_correction_mode}")
        print(f"conv_correction_channels  = {self.conv_correction_channels}")
        print(f"conv_kernel_size          = {self.conv_kernel_size}")
        print(f"conv_padding              = {self.conv_padding}")
        print(f"use_learnable_conv_scale  = {self.use_learnable_conv_scale}")
        print(f"conv_correction_scale_init= {self.conv_correction_scale_init}")
        print("compound r2r              = edge-aware GAT")
        print("compound i2i              = GCN")
        print("solvent branch            = edge-aware GAT")
        print("spectrum query            = graph-derived mol_feat only")
        print("split ablation            = enabled via spec_split_dims")
        print("final correction          = wavelength-local Conv1D correction head")
        print("loss                      = masked MAE")
        print("=" * 100)

        # ------------------------------------------------------------------
        # Embedding layers
        # ------------------------------------------------------------------
        self.embedding_rnode_lin = nn.Sequential(
            nn.Linear(config["node_dim"], hidden_dim, bias=False)
        )

        self.embedding_inode_lin = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=False)
        )

        # r2r edge is embedded before r2r GAT.
        # With current ISAGraphGenerator, edge_dim should be 1
        # because r2r edge feature is bond-order scalar.
        self.embedding_edge_lin = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim, bias=False)
        )

        # ------------------------------------------------------------------
        # Compound ISAT convolution
        # ------------------------------------------------------------------
        self.ISATconv = ISATconvolution(
            in_node_feats=hidden_dim,
            in_edge_feats=hidden_dim,
            out_feats=hidden_dim,
            activation=nn.LeakyReLU(self.gat_node_activation_slope),
            n_layers=gcn_layers,

            # r2r GAT dropout follows gat_dropout.
            gat_dropout=gat_dropout,

            # i2i GCN and d2score MLP dropout follow general dropout.
            gcn_dropout=dropout,
            score_dropout=dropout,

            batch_norm=False,
            residual_sum=False,

            # R3/R1 shared switches
            i2i_residual=self.i2i_residual,
            d2score_mode=self.d2score_mode,

            alpha=0.1,
            max_dist=d_edge_dim,
            heads=heads,
            mlp_activation=self.mlp_activation_name,
            gat_node_activation_slope=self.gat_node_activation_slope,
        )

        # ------------------------------------------------------------------
        # Solvent branch: edge-aware GAT
        # ------------------------------------------------------------------
        self.solvent_GAT = GATs(
            config["node_dim"],
            hidden_dim,
            hidden_dim,
            nn.LeakyReLU(self.gat_node_activation_slope),
            gcn_layers,
            gat_dropout,
            False,
            False,
            edge_feats=edge_dim,
            num_heads=heads,
        )

        # ------------------------------------------------------------------
        # Spectrum attention MLP blocks
        # ------------------------------------------------------------------
        self.mol_linear1 = Linears(hidden_dim, self.spec_dim, nn.Sigmoid(), 2, dropout, False)
        self.mol_linear2 = Linears(hidden_dim, self.spec_dim, nn.Sigmoid(), 2, dropout, False)
        self.mol_linear3 = Linears(hidden_dim, self.spec_dim, nn.Sigmoid(), 2, dropout, False)

        self.key_linear = Linears(hidden_dim * 2, self.spec_dim, nn.Sigmoid(), linear_layers, dropout, False)
        self.query_linear = Linears(self.spec_dim, self.spec_dim, nn.Sigmoid(), linear_layers, dropout, False)
        self.value_linear = Linears(hidden_dim * 2, hidden_dim, mlp_act, linear_layers, dropout, False)

        self.linear1 = Linears(self.spec_dim + hidden_dim, spec_dim1, mlp_act, linear_layers, dropout, False)
        self.linear2 = Linears(self.spec_dim + hidden_dim, spec_dim2, mlp_act, linear_layers, dropout, False)
        self.linear3 = Linears(self.spec_dim + hidden_dim, spec_dim3, mlp_act, linear_layers, dropout, False)

        # ------------------------------------------------------------------
        # Optional split final linear
        # ------------------------------------------------------------------
        if self.use_split_final_linear:
            self.final_linear1 = nn.Sequential(
                nn.Linear(spec_dim1, spec_dim1),
            )
            self.final_linear2 = nn.Sequential(
                nn.Linear(spec_dim2, spec_dim2),
            )
            self.final_linear3 = nn.Sequential(
                nn.Linear(spec_dim3, spec_dim3),
            )
        else:
            self.final_linear1 = None
            self.final_linear2 = None
            self.final_linear3 = None

        # ------------------------------------------------------------------
        # R3 final correction:
        #   wavelength-local Conv1D correction head
        # ------------------------------------------------------------------
        self.final_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=self.conv_correction_channels,
                kernel_size=self.conv_kernel_size,
                padding=self.conv_padding,
            ),
            nn.LeakyReLU(self.gat_node_activation_slope),
            nn.Dropout(dropout),

            nn.Conv1d(
                in_channels=self.conv_correction_channels,
                out_channels=self.conv_correction_channels,
                kernel_size=self.conv_kernel_size,
                padding=self.conv_padding,
            ),
            nn.LeakyReLU(self.gat_node_activation_slope),
            nn.Dropout(dropout),

            nn.Conv1d(
                in_channels=self.conv_correction_channels,
                out_channels=1,
                kernel_size=self.conv_kernel_size,
                padding=self.conv_padding,
            ),
        )

        # Alias intentionally disabled:
        #   R3_PM6FREE uses final_conv instead of dense final_linear.
        self.final_linear = None

        self.reduce = SumPooling()

        # Debug buffers.
        self.correction = None
        self.correction_raw = None
        self.last_mol_feat = None
        self.last_spectrum_att = None

    def _count_pm6_keys(self, kargs):
        """
        PM6-free diagnostic only.

        D4CMPP2 DataManager may still pass c330_var ... c1000_var
        because the dataset contains PM6 columns.

        This model intentionally ignores them.
        """
        count = 0

        if "pm6_spec" in kargs and kargs["pm6_spec"] is not None:
            count += 1

        for i in range(330, 1002, 2):
            if f"c{i}_var" in kargs:
                count += 1

        return count

    def _get_required(self, kargs, key):
        value = kargs.get(key)
        if value is None:
            raise KeyError(f"Missing required input key: {key}")
        return value

    def forward(self, **kargs):
        # ------------------------------------------------------------------
        # PM6-free diagnostic
        # ------------------------------------------------------------------
        self.used_pm6_spectrum = False
        self.ignored_pm6_key_count = self._count_pm6_keys(kargs)

        # ------------------------------------------------------------------
        # Compound ISAT inputs
        # ------------------------------------------------------------------
        graph = self._get_required(kargs, "compound_graphs")
        r_node = self._get_required(kargs, "compound_r_node")
        i_node = self._get_required(kargs, "compound_i_node")
        r_edge = self._get_required(kargs, "compound_r2r_edge")
        d_edge = self._get_required(kargs, "compound_d2d_edge")

        # ------------------------------------------------------------------
        # Solvent ISA graph inputs
        # ------------------------------------------------------------------
        solv_graph = self._get_required(kargs, "solvent_graphs")
        solv_r_node = self._get_required(kargs, "solvent_r_node")
        solv_r_edge = self._get_required(kargs, "solvent_r2r_edge")

        # ------------------------------------------------------------------
        # Compound embedding
        # ------------------------------------------------------------------
        r_node = self.embedding_rnode_lin(r_node.float())
        i_node = self.embedding_inode_lin(i_node.float())

        r_edge = self.embedding_edge_lin(r_edge.float())
        d_edge = d_edge.float()

        # ------------------------------------------------------------------
        # Compound ISAT convolution
        # ------------------------------------------------------------------
        real_graph = graph.node_type_subgraph(["r_nd"])
        real_graph.set_batch_num_nodes(graph.batch_num_nodes("r_nd"))
        real_graph.set_batch_num_edges(graph.batch_num_edges("r2r"))

        r_node, d_node = self.ISATconv(
            graph,
            r_node,
            r_edge,
            i_node,
            d_edge,
        )

        r_node = self.reduce(real_graph, r_node)

        dot_graph = graph.node_type_subgraph(["d_nd"])
        dot_graph.set_batch_num_nodes(graph.batch_num_nodes("d_nd"))
        dot_graph.set_batch_num_edges(graph.batch_num_edges("d2d"))

        # ------------------------------------------------------------------
        # Graph-derived spectral query template
        # ------------------------------------------------------------------
        # Spectra R3:
        #     spectrum_att = mol_feat * calc_spec
        #
        # PM6-free R3:
        #     spectrum_att = mol_feat
        #
        # Therefore, PM6 spectrum columns are intentionally not used.
        # ------------------------------------------------------------------
        mol_feat1 = self.mol_linear1(r_node)
        mol_feat2 = self.mol_linear2(r_node)
        mol_feat3 = self.mol_linear3(r_node)

        spectrum_att1 = mol_feat1
        spectrum_att2 = mol_feat2
        spectrum_att3 = mol_feat3

        self.last_mol_feat = (
            mol_feat1.detach(),
            mol_feat2.detach(),
            mol_feat3.detach(),
        )
        self.last_spectrum_att = (
            spectrum_att1.detach(),
            spectrum_att2.detach(),
            spectrum_att3.detach(),
        )

        # ------------------------------------------------------------------
        # Solvent branch: edge-aware GAT
        # ------------------------------------------------------------------
        solv_r_node = solv_r_node.float()
        solv_r_edge = solv_r_edge.float()

        solv_real_graph = solv_graph.node_type_subgraph(["r_nd"])
        solv_real_graph.set_batch_num_nodes(solv_graph.batch_num_nodes("r_nd"))
        solv_real_graph.set_batch_num_edges(solv_graph.batch_num_edges("r2r"))

        if solv_real_graph.num_edges() != solv_r_edge.shape[0]:
            raise ValueError(
                f"solvent r2r edge count mismatch: "
                f"graph edges={solv_real_graph.num_edges()}, "
                f"solvent_r2r_edge rows={solv_r_edge.shape[0]}"
            )

        solv_real_graph.edata["f"] = solv_r_edge

        solv_h = self.solvent_GAT(solv_real_graph, solv_r_node)
        solv_h = self.reduce(solv_real_graph, solv_h)

        # ------------------------------------------------------------------
        # Dot-node attention with solvent-conditioned d_node
        # ------------------------------------------------------------------
        d_batch_num_nodes = graph.batch_num_nodes("d_nd").to(d_node.device)

        solvent_tiled = solv_h.repeat_interleave(
            repeats=d_batch_num_nodes,
            dim=0,
        )

        d_solvent_cat = torch.cat([d_node, solvent_tiled], dim=1)

        k = self.key_linear(d_solvent_cat)

        q1 = self.query_linear(spectrum_att1)
        q2 = self.query_linear(spectrum_att2)
        q3 = self.query_linear(spectrum_att3)

        q1 = q1.repeat_interleave(repeats=d_batch_num_nodes, dim=0)
        q2 = q2.repeat_interleave(repeats=d_batch_num_nodes, dim=0)
        q3 = q3.repeat_interleave(repeats=d_batch_num_nodes, dim=0)

        v = self.value_linear(d_solvent_cat)

        scale = math.sqrt(float(self.spec_dim))

        att1 = torch.bmm(q1.unsqueeze(1), k.unsqueeze(2)).squeeze(-1).squeeze(-1) / scale
        att2 = torch.bmm(q2.unsqueeze(1), k.unsqueeze(2)).squeeze(-1).squeeze(-1) / scale
        att3 = torch.bmm(q3.unsqueeze(1), k.unsqueeze(2)).squeeze(-1).squeeze(-1) / scale

        dot_graph.ndata["att1"] = att1
        dot_graph.ndata["att2"] = att2
        dot_graph.ndata["att3"] = att3

        att1 = dgl.softmax_nodes(dot_graph, "att1")
        att2 = dgl.softmax_nodes(dot_graph, "att2")
        att3 = dgl.softmax_nodes(dot_graph, "att3")

        att_feat1 = att1.unsqueeze(1) * v
        att_feat2 = att2.unsqueeze(1) * v
        att_feat3 = att3.unsqueeze(1) * v

        att_feat1 = self.reduce(dot_graph, att_feat1)
        att_feat2 = self.reduce(dot_graph, att_feat2)
        att_feat3 = self.reduce(dot_graph, att_feat3)

        feat1 = torch.cat([spectrum_att1, att_feat1], dim=1)
        feat2 = torch.cat([spectrum_att2, att_feat2], dim=1)
        feat3 = torch.cat([spectrum_att3, att_feat3], dim=1)

        spec1 = self.linear1(feat1)
        spec2 = self.linear2(feat2)
        spec3 = self.linear3(feat3)

        if self.use_split_final_linear:
            spec1 = self.final_linear1(spec1)
            spec2 = self.final_linear2(spec2)
            spec3 = self.final_linear3(spec3)

        concat_spec = torch.cat([spec1, spec2, spec3], dim=1)

        if concat_spec.shape[1] != self.spec_dim:
            raise ValueError(
                f"concat_spec dimension mismatch: "
                f"got {concat_spec.shape[1]}, expected {self.spec_dim}"
            )

        # ------------------------------------------------------------------
        # R3 wavelength-local Conv1D correction
        # ------------------------------------------------------------------
        correction_raw = self.final_conv(concat_spec.unsqueeze(1)).squeeze(1)

        if correction_raw.shape != concat_spec.shape:
            raise ValueError(
                f"Conv1D correction shape mismatch: "
                f"correction_raw={correction_raw.shape}, "
                f"concat_spec={concat_spec.shape}"
            )

        correction = self.conv_correction_scale * correction_raw

        self.correction_raw = correction_raw
        self.correction = correction

        output = concat_spec + correction

        return output

    def loss_fn(self, scores, targets):
        """
        PM6-free R3 loss:
            masked MAE only
        """
        mask = ~torch.isnan(targets)

        if mask.sum() == 0:
            return torch.tensor(
                0.0,
                dtype=scores.dtype,
                device=scores.device,
                requires_grad=True,
            )

        return torch.mean(torch.abs(targets[mask] - scores[mask]))


class ISATconvolution(nn.Module):
    def __init__(
        self,
        in_node_feats,
        in_edge_feats,
        out_feats,
        activation,
        n_layers,
        gat_dropout=0.0,
        gcn_dropout=0.1,
        score_dropout=0.1,
        batch_norm=False,
        residual_sum=False,
        i2i_residual=True,
        d2score_mode="vector_gate",
        alpha=0.1,
        max_dist=4,
        heads=8,
        mlp_activation="leakyrelu",
        gat_node_activation_slope=0.1,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.i2i_residual = i2i_residual
        self.d2score_mode = d2score_mode

        # ------------------------------------------------------------------
        # r2r message passing:
        # edge-aware GATs
        # ------------------------------------------------------------------
        self.r2r = GATs(
            in_node_feats,
            out_feats,
            out_feats,
            activation,
            n_layers,
            gat_dropout,
            batch_norm,
            residual_sum,
            edge_feats=in_edge_feats,
            num_heads=heads,
        )

        # ------------------------------------------------------------------
        # i2i remains original GCN stack
        # R3/R1 default: i2i_residual=True
        # ------------------------------------------------------------------
        self.i2i = nn.ModuleList([
            GCN_layer(
                out_feats,
                out_feats,
                activation,
                gcn_dropout,
                batch_norm,
                i2i_residual,
            )
            for _ in range(n_layers)
        ])

        self.r2i = r2i_layer()
        self.i2d = i2s_layer()
        self.d2d = distGCN_layer(out_feats, max_dist, out_feats, activation, alpha)

        # ------------------------------------------------------------------
        # d2score
        # ------------------------------------------------------------------
        if d2score_mode == "vector_gate":
            score_act = get_activation(mlp_activation, gat_node_activation_slope)

            self.d2score = Linears(
                out_feats,
                out_feats,
                score_act,
                2,
                score_dropout,
                False,
            )
            self.score_bn = nn.BatchNorm1d(out_feats)

        elif d2score_mode == "scalar_gate":
            self.d2score = nn.Sequential(
                nn.Linear(out_feats, out_feats // 2),
                nn.LeakyReLU(),
                nn.Linear(out_feats // 2, 1),
            )
            self.score_bn = nn.BatchNorm1d(1)

        else:
            raise ValueError(
                f"d2score_mode must be 'vector_gate' or 'scalar_gate', "
                f"got {d2score_mode}"
            )

        self.d2r = s2r_Layer()
        self.reduce = SumPooling()

    def forward(self, graph, r_node, r2r_edge, i_node, d2d_edge):
        real_graph = graph.node_type_subgraph(["r_nd"])
        real_graph.set_batch_num_nodes(graph.batch_num_nodes("r_nd"))
        real_graph.set_batch_num_edges(graph.batch_num_edges("r2r"))

        image_graph = graph.node_type_subgraph(["i_nd"])
        image_graph.set_batch_num_nodes(graph.batch_num_nodes("i_nd"))
        image_graph.set_batch_num_edges(graph.batch_num_edges("i2i"))

        dot_graph = graph.node_type_subgraph(["d_nd"])
        dot_graph.set_batch_num_nodes(graph.batch_num_nodes("d_nd"))
        dot_graph.set_batch_num_edges(graph.batch_num_edges("d2d"))

        if real_graph.num_edges() != r2r_edge.shape[0]:
            raise ValueError(
                f"compound r2r edge count mismatch: "
                f"real_graph edges={real_graph.num_edges()}, "
                f"r2r_edge rows={r2r_edge.shape[0]}"
            )

        real_graph.edata["f"] = r2r_edge

        # Keep original ISAT order:
        # r2r -> r2i -> i2i, repeated n_layers times.
        for i in range(self.n_layers):
            r_node = self.r2r.layers[i](real_graph, r_node)

            i_node = self.r2i(graph, r_node, i_node)
            i_node = self.i2i[i](image_graph, i_node)

        d_node = self.i2d(graph, i_node)
        d_node = self.d2d(dot_graph, d_node, d2d_edge)

        score = self.d2r(graph, d_node)
        score = self.d2score(score)
        score = self.score_bn(score)
        score = torch.sigmoid(score)

        r_node = score * r_node

        return r_node, d_node


class r2i_layer(nn.Module):
    def forward(self, graph, r_node, i_node):
        return i_node + r_node


class i2s_layer(nn.Module):
    def forward(self, graph, i_node):
        with graph.local_scope():
            graph = graph.edge_type_subgraph([("i_nd", "i2d", "d_nd")])
            graph.nodes["i_nd"].data["h"] = i_node
            graph.update_all(
                dgl.function.copy_u("h", "mail"),
                dgl.function.sum("mail", "h"),
            )
            d_node = graph.nodes["d_nd"].data["h"]

        return d_node


class s2r_Layer(nn.Module):
    def forward(self, graph, node):
        with graph.local_scope():
            graph = graph.edge_type_subgraph([("d_nd", "d2r", "r_nd")])
            graph.nodes["d_nd"].data["h"] = node
            graph.update_all(
                dgl.function.copy_u("h", "mail"),
                dgl.function.sum("mail", "h"),
            )
            score = graph.nodes["r_nd"].data["h"]

        return score