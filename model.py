from typing import Any

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from dgl.nn import EGATConv
from pytorch_lightning.core import LightningModule
from torchmetrics import Accuracy
from EdgeGat import EdgeGraphAttention

class GraphEGATDGL(nn.Module):
    def __init__(
        self,
        in_node_feats,
        in_edge_feats,
        h_node_feats,
        h_edge_feats,
        num_heads,
        num_classes_node,
        num_classes_edge
    ) -> None:
        super().__init__()
        self.conv1 = EGATConv(
            in_node_feats, in_edge_feats, h_node_feats, h_edge_feats, num_heads
        )
        self.conv2 = EGATConv(
            h_node_feats * num_heads,
            h_edge_feats * num_heads,
            h_node_feats,
            h_edge_feats,
            num_heads,
        )
        self.conv3 = EGATConv(
            h_node_feats * num_heads,
            h_edge_feats * num_heads,
            h_node_feats,
            h_edge_feats,
            num_heads,
        )
        self.conv4 = EGATConv(
            h_node_feats * num_heads,
            h_edge_feats * num_heads,
            h_node_feats,
            h_edge_feats,
            num_heads,
        )

        # Classification heads
        self.node_classifier = torch.nn.Linear(h_node_feats * num_heads, num_classes_node)
        self.edge_classifier = torch.nn.Linear(h_edge_feats * num_heads, num_classes_edge)
        self.link_classifier = nn.Sequential(
            torch.nn.Linear(h_node_feats * num_heads*2 + h_edge_feats * num_heads, h_edge_feats * num_heads),
            torch.nn.ReLU(),
            torch.nn.Linear(h_edge_feats * num_heads, 1),
        )

    def apply_edges(self, edges):
        # linear_2 ( relu ( linear_1 ( [h_u, h_v] ) ) )
        h = torch.cat([edges.src["h"], edges.dst["h"], edges.data["h"]], 1)
        return {"score": self.link_classifier(h)}

    def link_predictor(self, g, h_nodes, h_edges):
        with g.local_scope():
            g.ndata["h"] = h_nodes
            g.edata["h"] = h_edges
            g.apply_edges(self.apply_edges)
            return g.edata["score"]

    def forward(self, graph, node_feature, edge_feature):
        h_node, h_edge = self.conv1(graph, node_feature, edge_feature)
        h_node = F.relu(h_node)
        h_edge = F.relu(h_edge)
        
        h_node, h_edge = self.conv2(
            graph, h_node.flatten(1, 2), h_edge.flatten(1, 2)
        )
        h_node = F.relu(h_node)
        h_edge = F.relu(h_edge)

        h_node, h_edge = self.conv3(
            graph, h_node.flatten(1, 2), h_edge.flatten(1, 2)
        )
        h_node = F.relu(h_node)
        h_edge = F.relu(h_edge)
        
        h_node, h_edge = self.conv4(
            graph, h_node.flatten(1, 2), h_edge.flatten(1, 2)
        )

        # Node and edge classification
        node_logits = self.node_classifier(h_node.flatten(1, 2))
        edge_logits = self.edge_classifier(h_edge.flatten(1, 2))
        link_logits = self.link_predictor(graph, h_node.flatten(1, 2), h_edge.flatten(1, 2))
        
        return node_logits, edge_logits, link_logits



class GNNTrainer(LightningModule):
    def __init__(
        self,
        in_node_feats: int = 101,
        in_edge_feats: int = 7,
        h_node_feats: int = 32,
        h_edge_feats: int = 32,
        num_heads: int = 4,
        num_symbols: int = 101,
        num_relations: int = 7,
        lr: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.model = GraphEGATDGL(
            in_node_feats, in_edge_feats, h_node_feats, h_edge_feats, num_heads, num_symbols, num_relations
        )
        self.sym_acc = Accuracy(task='multiclass', num_classes=num_symbols)
        self.rel_acc = Accuracy(task='multiclass', num_classes=num_relations)
        self.in_node_feats = in_node_feats
        self.in_edge_feats = in_edge_feats
        self.norel_idx = in_edge_feats - 1

    def forward(self, graph, node_feature, edge_feature):
        self.model.forward(graph, node_feature, edge_feature)

    def process_graph(self, graph):
        num_edges = graph.num_edges()
        # bidirectional
        graph = dgl.add_self_loop(graph, fill_data=0.0)
        num_added_edges = graph.num_edges() - num_edges
        
        node_logits, edge_logits, link_logits = self.model.forward(
            graph,
            F.one_hot(
                graph.ndata["input_syms"], num_classes=self.in_node_feats
            ).float(),
            F.one_hot(
                graph.edata["input_rels"], num_classes=self.in_edge_feats
            ).float()
            * graph.edata["input_relprobs"].unsqueeze(-1).float(),
        )

        link_logits = link_logits[:-num_added_edges]
        link_labels = (graph.edata["label_rels"][:-num_added_edges] != self.norel_idx).float().unsqueeze(-1)

        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)

        link_acc = torch.sum((link_logits > 0) == link_labels).float() / link_labels.shape[0]

        accuracy = link_acc == 1.0

        return loss, link_acc, accuracy

    def training_step(self, batch, batch_idx):
        train_g = batch
        # if train_g.num_edges() > 20:
        #     train_g = dgl.DropEdge(p=0.05)(train_g)

        loss, link_acc, accuracy = self.process_graph(train_g)
        
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        
        self.log(
            "train_link_acc",
            link_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )

        self.log(
            "train_seq_acc",
            accuracy,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )

        return loss

    def eval_step(self, batch, batch_idx, prefix: str):
        val_g = batch
        loss, link_acc, accuracy = self.process_graph(val_g)

        self.log(
            f"{prefix}_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )

        self.log(
            f"{prefix}_link_acc",
            link_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

        self.log(
            f"{prefix}_seq_acc",
            accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        
        return loss

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 0.1 ** (epoch // 20)
        )
        return [optimizer], [scheduler]


        