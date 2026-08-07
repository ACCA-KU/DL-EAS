import torch
import torch.nn as nn
from dgl.nn.functional import edge_softmax


class GAT_layer(nn.Module):
    """
    Edge-aware multi-head GAT layer.

    attention_ij = source term + destination term + edge term
    multi-head outputs are averaged, not concatenated.
    No residual connection by default.
    """

    def __init__(
        self,
        in_node_feats,
        hidden_feats,
        out_feats,
        activation,
        dropout=0.2,
        batch_norm=False,
        residual_sum=False,
        edge_feats=1,
        num_heads=8,
    ):
        super(GAT_layer, self).__init__()

        self.in_node_feats = in_node_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.residual_sum = residual_sum
        self.edge_feats = edge_feats
        self.num_heads = num_heads

        self.W_node = nn.Linear(in_node_feats, out_feats * num_heads, bias=False)
        self.W_edge = nn.Linear(edge_feats, num_heads, bias=False)

        self.att_src = nn.Parameter(torch.empty(num_heads, out_feats))
        self.att_dst = nn.Parameter(torch.empty(num_heads, out_feats))

        self.att_act = nn.LeakyReLU(0.2)

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_feats)

        if self.residual_sum:
            if in_node_feats != out_feats:
                self.residual_layer = nn.Linear(in_node_feats, out_feats)
            else:
                self.residual_layer = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_node.weight)
        nn.init.xavier_uniform_(self.W_edge.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def edge_attention(self, edges):
        src = edges.src["h_trans"].view(-1, self.num_heads, self.out_feats)
        dst = edges.dst["h_trans"].view(-1, self.num_heads, self.out_feats)
        e = edges.data["e_trans"].view(-1, self.num_heads, 1)

        attn = (
            (src * self.att_src).sum(dim=-1)
            + (dst * self.att_dst).sum(dim=-1)
            + e.squeeze(-1)
        )

        return {"e_attn": self.att_act(attn)}

    def message_func(self, edges):
        src = edges.src["h_trans"].view(-1, self.num_heads, self.out_feats)
        alpha = edges.data["alpha"].unsqueeze(-1)
        return {"m": src * alpha}

    def reduce_func(self, nodes):
        h = nodes.mailbox["m"].sum(dim=1)   # (N, heads, out_feats)
        h = h.mean(dim=1)                   # (N, out_feats)
        return {"h_new": h}

    def forward(self, graph, node_feats):
        with graph.local_scope():
            h_trans = self.W_node(node_feats)
            graph.ndata["h_trans"] = h_trans

            if graph.num_edges() == 0:
                h = h_trans.view(-1, self.num_heads, self.out_feats).mean(dim=1)

                if self.batch_norm:
                    h = self.bn(h)

                if self.activation is not None:
                    h = self.activation(h)

                h = self.dropout(h)

                if self.residual_sum:
                    h = h + self.residual_layer(node_feats)

                return h

            if "f" not in graph.edata:
                raise KeyError(
                    "GAT_layer expects graph.edata['f'] as edge features."
                )

            edge_feats = graph.edata["f"]
            if edge_feats.dim() == 1:
                edge_feats = edge_feats.view(-1, 1)

            if edge_feats.shape[1] != self.edge_feats:
                raise ValueError(
                    f"Edge feature dimension mismatch: "
                    f"graph.edata['f'].shape[1]={edge_feats.shape[1]}, "
                    f"but this GAT_layer was initialized with edge_feats={self.edge_feats}."
                )

            graph.edata["e_trans"] = self.W_edge(edge_feats)

            graph.apply_edges(self.edge_attention)
            graph.edata["alpha"] = edge_softmax(graph, graph.edata["e_attn"])

            graph.update_all(self.message_func, self.reduce_func)

            h = graph.ndata["h_new"]

            if self.batch_norm:
                h = self.bn(h)

            if self.activation is not None:
                h = self.activation(h)

            h = self.dropout(h)

            if self.residual_sum:
                h = h + self.residual_layer(node_feats)

            return h


class GATs(nn.Module):
    """
    Stack of edge-aware multi-head GAT layers.
    """

    def __init__(
        self,
        in_node_feats,
        hidden_feats,
        out_feats,
        activation,
        n_layers,
        dropout=0.2,
        batch_norm=False,
        residual_sum=False,
        edge_feats=1,
        num_heads=8,
    ):
        super(GATs, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(n_layers):
            if i == 0:
                _in_feats = in_node_feats
            else:
                _in_feats = hidden_feats

            if i == n_layers - 1:
                _out_feats = out_feats
            else:
                _out_feats = hidden_feats

            self.layers.append(
                GAT_layer(
                    in_node_feats=_in_feats,
                    hidden_feats=hidden_feats,
                    out_feats=_out_feats,
                    activation=activation,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    residual_sum=residual_sum,
                    edge_feats=edge_feats,
                    num_heads=num_heads,
                )
            )

    def forward(self, graph, node_feats):
        h = node_feats
        for layer in self.layers:
            h = layer(graph, h)
        return h