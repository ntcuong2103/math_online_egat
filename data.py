#!/usr/bin/env python
import json
import os
import pickle
import shutil
from glob import glob
from typing import Optional, Tuple, Dict, List
from collections import defaultdict

import dgl
import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import networkx as nx

from parse_lg import parse_lg
from utils.los import LOS


##############################################################################
# Mapping dictionaries used to convert between LG and JSON relation names
##############################################################################
map_lg_to_json = {
    "Right": "R",
    "Above": "Ab",
    "Below": "Be",
    "Inside": "In",
    "Sub": "Sub",
    "Sup": "Sup",
}

# For symbol conversion in the CROHME dataset
map_lg_to_json_sym = {
    "\\frac": "-",
    "<": "\\lt",
    ">": "\\gt",
}


##############################################################################
# Basic Classes for Symbol and Relation (used when reading JSON or LG files)
##############################################################################
class SymbolCandidate:
    __slots__ = ("latex", "strokeIds")

    def __init__(self, *args):
        if len(args) == 2:
            self.latex = args[0]
            self.strokeIds = args[1]

    def toJson(self):
        return {"latex": self.latex, "strokeIds": self.strokeIds}


class Relation:
    __slots__ = ("prob", "relation", "symChild", "symRoot")

    def __init__(self, *args):
        if len(args) == 4:
            self.symRoot = args[0]
            self.symChild = args[1]
            self.relation = args[2]
            self.prob = args[3]

    def toJson(self):
        return {
            "prob": self.prob,
            "relation": self.relation,
            "symChild": self.symChild.toJson(),
            "symRoot": self.symRoot.toJson(),
        }


##############################################################################
# Load vocabulary (each line in HandsCTC.lbl)
##############################################################################
vocab = [line.strip() for line in open("data/HandsCTC.lbl").readlines()]
sym_vocab = vocab[:-7]
rel_vocab = vocab[-7:]


##############################################################################
# LINE-OF-SIGHT (LOS) FUNCTIONALITY
##############################################################################

import xml.etree.ElementTree as ET

import numpy as np


class Segment(object):
    """Class to reprsent a Segment compound of strokes (id) with an id and label."""

    __slots__ = ("id", "label", "strId")

    def __init__(self, *args):
        if len(args) == 3:
            self.id = args[0]
            self.label = args[1]
            self.strId = args[2]
        else:
            self.id = "none"
            self.label = ""
            self.strId = set([])


class Inkml(object):
    """Class to represent an INKML file with strokes, segmentation and labels"""

    __slots__ = ("fileName", "strokes", "strkOrder", "segments", "truth", "UI")

    NS = {
        "ns": "http://www.w3.org/2003/InkML",
        "xml": "http://www.w3.org/XML/1998/namespace",
    }

    def __init__(self, *args):
        self.fileName = None
        self.strokes = {}
        self.strkOrder = []
        self.segments = {}
        self.truth = ""
        self.UI = ""
        if len(args) == 1:
            self.fileName = args[0]
            self.loadFromFile()

    def fixNS(self, ns, att):
        """Build the right tag or element name with namespace"""
        return "{" + Inkml.NS[ns] + "}" + att

    def loadFromFile(self):
        """load the ink from an inkml file (strokes, segments, labels)"""
        tree = ET.parse(self.fileName)
        # # ET.register_namespace();
        root = tree.getroot()
        for info in root.findall("ns:annotation", namespaces=Inkml.NS):
            if "type" in info.attrib:
                if info.attrib["type"] == "truth":
                    self.truth = info.text.strip()
                if info.attrib["type"] == "UI":
                    self.UI = info.text.strip()
        for strk in root.findall("ns:trace", namespaces=Inkml.NS):
            self.strokes[strk.attrib["id"]] = strk.text.strip()
            self.strkOrder.append(strk.attrib["id"])
        segments = root.find("ns:traceGroup", namespaces=Inkml.NS)
        if segments is None or len(segments) == 0:
            return
        for seg in segments.iterfind("ns:traceGroup", namespaces=Inkml.NS):
            id = seg.attrib[self.fixNS("xml", "id")]
            label = seg.find("ns:annotation", namespaces=Inkml.NS).text
            strkList = set([])
            for t in seg.findall("ns:traceView", namespaces=Inkml.NS):
                strkList.add(t.attrib["traceDataRef"])
            self.segments[id] = Segment(id, label, strkList)

    def getTraces(self, height=256):
        traces_array = [
            np.array(
                [
                    p.strip().split()[:2]  # only get two channels
                    for p in self.strokes[id].split(",")
                ],
                dtype="float",
            )
            for id in self.strkOrder
        ]

        ratio = height / (
            (
                np.concatenate(traces_array, 0).max(0)
                - np.concatenate(traces_array, 0).min(0)
            )[1]
            + 1e-6
        )
        return [(trace * ratio).astype(int).tolist() for trace in traces_array]


def generate_dummy_stroke_points(
    stroke_ids: List[int],
) -> List[List[Tuple[float, float]]]:
    """
    Generate dummy stroke points for a given list of stroke IDs.
    For each stroke id, we create a short horizontal segment.
    """
    stroke_points = []
    base_y = 100  # Base y-position for strokes
    for s in stroke_ids:
        x = s * 10
        # Add a small vertical variation so not all strokes are exactly aligned
        y = base_y + (s % 3) * 5
        stroke_points.append([(x, y), (x + 10, y)])
    return stroke_points


class BoundingBox:
    def __init__(self, stroke_points: List[List[Tuple[float, float]]]):
        # Flatten the list of points and compute min/max
        x_coords = [p[0] for stroke in stroke_points for p in stroke]
        y_coords = [p[1] for stroke in stroke_points for p in stroke]
        self.x_min = min(x_coords)
        self.x_max = max(x_coords)
        self.y_min = min(y_coords)
        self.y_max = max(y_coords)
        self.center_x = (self.x_min + self.x_max) / 2
        self.center_y = (self.y_min + self.y_max) / 2

    def overlaps_horizontally(self, other: "BoundingBox") -> bool:
        return not (self.x_max < other.x_min or other.x_max < self.x_min)

    def overlaps_vertically(self, other: "BoundingBox") -> bool:
        return not (self.y_max < other.y_min or other.y_max < self.y_min)

    def get_vertical_overlap(self, other: "BoundingBox") -> float:
        if not self.overlaps_vertically(other):
            return 0.0
        overlap = min(self.y_max, other.y_max) - max(self.y_min, other.y_min)
        my_height = self.y_max - self.y_min
        return overlap / my_height if my_height > 0 else 0.0



def load_json_file_with_los(json_path: str, inkml_path: str) -> Tuple[Dict, List]:
    """
    Load the JSON file containing the graph and apply LOS filtering.
    Returns:
      nodes: a dict mapping a symbol's stroke IDs (as string) to its label.
      filtered_edges: a list of tuples (parent, child, relation, prob) that passed LOS.
    """
    with open(json_path, "r") as f:
        relations = json.load(f)

    nodes = {}
    edges = []

    # Load the INKML file to get the stroke points
    inkml = Inkml(inkml_path)
    strokes = inkml.getTraces()

    # Create symbol-level stroke mapping
    symbol_strokes = {}
    for rel in relations:
        # Process the "root" symbol
        root_strokes = rel["symRoot"]["strokeIds"]
        root_latex = rel["symRoot"]["latex"]
        root_symbol_strokes = [strokes[idx] for idx in root_strokes]
        root_strokes_str = str(root_strokes)

        nodes[root_strokes_str] = {"label": root_latex}
        symbol_strokes[root_strokes_str] = root_symbol_strokes

        # Process the "child" symbol
        child_strokes = rel["symChild"]["strokeIds"]
        child_latex = rel["symChild"]["latex"]
        child_symbol_strokes = [strokes[idx] for idx in child_strokes]
        child_strokes_str = str(child_strokes)

        nodes[child_strokes_str] = {"label": child_latex}
        symbol_strokes[child_strokes_str] = child_symbol_strokes

        edges.append(
            (root_strokes_str, child_strokes_str, rel["relation"], rel["prob"])
        )

    # Merge strokes for each symbol before LOS
    unique_symbols = list(symbol_strokes.keys())
    merged_strokes = [
        [point for stroke in symbol_strokes[symbol] for point in stroke]
        for symbol in unique_symbols
    ]

    # Compute LOS adjacency matrix
    los_matrix = LOS(merged_strokes)

    # Filter edges based on LOS
    filtered_edges = []
    seen_pairs = set()
    sorted_edges = sorted(edges, key=lambda x: x[3], reverse=True)

    for idx, edge in enumerate(sorted_edges):
        parent, child, relation, prob = edge
        if (parent, child) in seen_pairs:
            continue

        parent_idx = unique_symbols.index(parent)
        child_idx = unique_symbols.index(child)

        if relation in ["R", "Ab", "Be", "Sub", "Sup", "In"]:
            if los_matrix[parent_idx][child_idx]:
                filtered_edges.append((parent, child, relation, prob))
                seen_pairs.add((parent, child))
        else:
            filtered_edges.append((parent, child, relation, prob))

    return nodes, filtered_edges


##############################################################################
# DATA LOADING & GRAPH CREATION FUNCTIONS
##############################################################################

class GraphData:
    def __init__(self):
        self.Symbols = {}
        self.Relations = {}

    def readJson(self, filename):
        relations = json.load(open(filename))
        for rel in relations:
            self.Symbols.update(
                {str(rel["symChild"]["strokeIds"]): {"label": rel["symChild"]["latex"]}}
            )
            self.Symbols.update(
                {str(rel["symRoot"]["strokeIds"]): {"label": rel["symRoot"]["latex"]}}
            )
            self.Relations.update(
                {
                    str([rel["symRoot"]["strokeIds"], rel["symChild"]["strokeIds"]]): {
                        "relation": rel["relation"],
                        "prob": rel["prob"],
                    }
                }
            )

    def readLGs(self, filename):
        objs, rels = parse_lg(filename)
        for obj in objs:
            self.Symbols.update({str(obj.strokes): {"label": obj.sym}})
        id2strokes = {obj.id: obj.strokes for obj in objs}
        for rel in rels:
            if rel.id1 not in id2strokes or rel.id2 not in id2strokes:
                print(
                    f"check {filename}: {rel.id1}, {rel.id2} not in {id2strokes.keys()}"
                )
                continue
            self.Relations.update(
                {
                    str([id2strokes[rel.id1], id2strokes[rel.id2]]): {
                        "relation": map_lg_to_json[rel.relation],
                        "prob": rel.prob,
                    }
                }
            )

    def filterNodes(self, nodes: List[str]):
        self.Symbols = {node: self.Symbols[node] for node in nodes}
        self.Relations = {
            rel: self.Relations[rel]
            for rel in self.Relations
            if str(eval(rel)[0]) in nodes and str(eval(rel)[1]) in nodes
        }

    def filterEdges(self, edges: List[str]):
        self.Relations = {
            rel: self.Relations[rel] for rel in edges if rel in self.Relations
        }

    def getEdgesByLOS(self, inkmlfile: str):
        # Load the INKML file to get the stroke points
        inkml = Inkml(inkmlfile)
        strokes = inkml.getTraces()

        # Merge strokes for each symbol before LOS
        merged_strokes = [
            [point for stroke in [strokes[i] for i in eval(symbol)] for point in stroke]
            for symbol in self.Symbols
        ]

        # Compute LOS adjacency matrix
        los_matrix = LOS(merged_strokes)

        # set of edges from los
        los_edges = {}
        for i, symbol1 in enumerate(self.Symbols):
            for j, symbol2 in enumerate(self.Symbols):
                if i == j:
                    continue
                if los_matrix[i][j]:
                    los_edges.update(
                        {
                            str([eval(symbol1), eval(symbol2)]): self.Relations.get(
                                str([eval(symbol1), eval(symbol2)]),
                                {"relation": "No", "prob": 1.0},
                            )
                        }
                    )

        return los_edges

def get_common_graph(graph1, graph2):
    common_nodes = [
        node for node in graph1.Symbols.keys() if node in graph2.Symbols.keys()
    ]
    graph1.filterNodes(common_nodes)
    graph2.filterNodes(common_nodes)
    return graph1, graph2, common_nodes


def create_graph_dgl(input_json, input_lg, input_inkml, mode="JSON_LOS"):
    """
    Create a DGL graph from a JSON file and a corresponding LG file.
    If use_los==True, the JSON is loaded with line-of-sight filtering.
    """
    graph_json = GraphData()
    graph_json.readJson(input_json)
    if mode == 'JSON_ONLY':
        pass    
    elif mode == 'JSON_LOS':
        graph_json.filterEdges(graph_json.getEdgesByLOS(input_inkml))
    elif mode == 'LOS_ONLY':
        graph_json.Relations = graph_json.getEdgesByLOS(input_inkml)

    # Read LG file (ground truth) as before.
    graph_lg = GraphData()
    graph_lg.readLGs(input_lg)

    # Only keep the common nodes between the two graphs.
    graph_json, graph_lg, common_nodes = get_common_graph(graph_json, graph_lg)
    node2id = {node: idx for idx, node in enumerate(common_nodes)}

    # Build the DGL graph (using the JSON edge definitions)
    g = dgl.graph(
        (
            [node2id[str(eval(rel)[0])] for rel in graph_json.Relations],
            [node2id[str(eval(rel)[1])] for rel in graph_json.Relations],
        ),
        num_nodes=len(common_nodes),
    )
    # NOTE: For the JSON graph, each symbol is stored as a dict with key "label"
    g.ndata["input_syms"] = torch.from_numpy(
        np.array(
            [
                sym_vocab.index(graph_json.Symbols[node]["label"])
                for node in common_nodes
            ]
        )
    )
    g.edata["input_rels"] = torch.from_numpy(
        np.array(
            [
                rel_vocab.index(graph_json.Relations[rel]["relation"])
                for rel in graph_json.Relations
            ]
        )
    )
    g.edata["input_relprobs"] = torch.from_numpy(
        np.array([graph_json.Relations[rel]["prob"] for rel in graph_json.Relations])
    )
    g.ndata["label_syms"] = torch.from_numpy(
        np.array(
            [
                sym_vocab.index(
                    map_lg_to_json_sym.get(
                        graph_lg.Symbols[node]["label"], graph_lg.Symbols[node]["label"]
                    )
                )
                for node in common_nodes
            ]
        )
    )
    g.edata["label_rels"] = torch.from_numpy(
        np.array(
            [
                rel_vocab.index(
                    graph_lg.Relations.get(rel, {"relation": "No"})["relation"]
                )
                for rel in graph_json.Relations
            ]
        )
    )
    return g

##############################################################################
# (Optional) Functions for visualizing the graphs as DOT files
##############################################################################
def load_graph_json_to_nx(input_json):
    try:
        relations = json.load(open(input_json))
    except Exception as e:
        return None, None
    if len(relations) == 0:
        return None, None
    sym2Latex = {}
    for rel in relations:
        sym2Latex.update({str(rel["symChild"]["strokeIds"]): rel["symChild"]["latex"]})
        sym2Latex.update({str(rel["symRoot"]["strokeIds"]): rel["symRoot"]["latex"]})
    edges = [
        (
            str(rel["symRoot"]["strokeIds"]),
            str(rel["symChild"]["strokeIds"]),
            {"weight": rel["prob"], "relation": rel["relation"]},
        )
        for rel in relations
    ]
    G = nx.MultiDiGraph()
    G.add_edges_from(
        [
            (
                str(rel["symRoot"]["strokeIds"]),
                str(rel["symChild"]["strokeIds"]),
                {"weight": rel["prob"], "relation": rel["relation"]},
            )
            for rel in relations
        ]
    )
    for node_id in G.nodes:
        G.nodes[node_id]["label"] = sym2Latex[node_id]
    return G, {node: idx for idx, node in enumerate(G.nodes)}


def load_lg_to_nx(input_lg):
    objs, rels = parse_lg(input_lg)
    G = nx.MultiDiGraph()
    id2strokes = {obj.id: str(obj.strokes) for obj in objs}
    strokes2obj = {str(obj.strokes): obj for obj in objs}
    for rel in rels:
        G.add_edge(
            id2strokes[rel.id1],
            id2strokes[rel.id2],
            relation=map_lg_to_json[rel.relation],
            weight=rel.prob,
        )
    for node_id in G.nodes:
        G.nodes[node_id]["label"] = strokes2obj[node_id].sym
    return G


def write_dot(G, output_fn):
    dot_str = []
    dot_str += [
        "strict digraph G {",
        "    rankdir=LR;",
        "    node [shape=record, width=.1]",
    ]
    dot_str += [
        f'    "{node}" [ label="{G.nodes()[node]["label"]}"];' for node in G.nodes()
    ]
    for edge in G.edges():
        rel, prob = (
            G.get_edge_data(*edge)[0]["relation"],
            G.get_edge_data(*edge)[0]["weight"],
        )
        if rel in ["Ab", "Sup"]:
            dot_str += [
                f'    "{edge[0]}":ne -> "{edge[1]}" [label="{rel}", prob={prob}];'
            ]
        elif rel in ["Be", "In", "Sub"]:
            dot_str += [
                f'    "{edge[0]}":se -> "{edge[1]}" [label="{rel}", prob={prob}];'
            ]
        elif rel in ["R", "No"]:
            dot_str += [
                f'    "{edge[0]}" -> "{edge[1]}" [label="{rel}", weight=2, prob={prob}];'
            ]
    dot_str += ["}"]
    with open(output_fn, "w") as f:
        f.writelines([line + "\n" for line in dot_str])


def visualize_json_lg(input_json, input_lg):
    G, _ = load_graph_json_to_nx(input_json)
    if G is None:
        return None
    G_lg = load_lg_to_nx(input_lg)
    if G_lg is None:
        return None
    write_dot(G, "graph.dot")
    write_dot(G_lg, "graph_lg.dot")


##############################################################################
# DATASET CLASSES (for training, etc.)
##############################################################################
class MathGraph(DGLDataset):
    def __init__(self):
        self.graphs = []
        super().__init__(name="crohme2019")

    def process(self):
        return

    def create_data(self, input_jsons, input_lgs, input_inkmls, use_los: bool = False):
        for fn in glob(f"{input_jsons}/*.json"):
            fn_lg = os.path.join(
                input_lgs, os.path.basename(fn).replace(".json", ".lg")
            )
            fn_inkml = os.path.join(
                input_inkmls, os.path.basename(fn).replace(".json", ".inkml")
            )
            try:
                g = create_graph_dgl(fn, fn_lg, fn_inkml, use_los=use_los)
                if g is not None and g.num_nodes() > 0 and g.num_edges() > 0:
                    self.graphs.append(g)
                else:
                    print(
                        f"Skipping graph: {fn}, num_nodes: {g.num_nodes()}, num_edges: {g.num_edges()}"
                    )
            except Exception as e:
                print(f"Error processing {fn}: {e}")

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)


def get_dataset(pkl_fn):
    graph_data = pickle.load(open(pkl_fn, "rb"))
    graph_ds = MathGraph()
    graph_ds.graphs = graph_data
    return graph_ds


import pytorch_lightning as pl


class MathGraphData(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        workers: int = 1,
        train_data: str = "train.pkl",
        val_data: str = "test2014.pkl",
        test_data: str = "test2019.pkl",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = get_dataset(self.train_data)
            self.val_dataset = get_dataset(self.val_data)
        if stage == "test" or stage is None:
            self.test_dataset = get_dataset(self.test_data)

    def train_dataloader(self):
        train_loader = GraphDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = GraphDataLoader(
            self.val_dataset, batch_size=self.batch_size, drop_last=False, shuffle=False
        )
        return val_loader

    def test_dataloader(self):
        test_loader = GraphDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )
        return test_loader


##############################################################################
# MAIN ENTRY POINT
##############################################################################
if __name__ == "__main__":
    # test create graph
    input_json = "data/Test2014_primitive_json/18_em_0.json"
    input_lg = "data/Crohme_all_LGs/18_em_0.lg"
    input_inkml = "data/Test2014/18_em_0.inkml"
    g = create_graph_dgl(input_json, input_lg, input_inkml, use_los=True)

    exit(0)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsons",
        type=str,
        required=True,
        help="Path to the folder containing input JSON files",
    )
    parser.add_argument(
        "--input_lgs",
        type=str,
        required=True,
        help="Path to the folder containing input LG files",
    )
    parser.add_argument(
        "--input_inkmls",
        type=str,
        default="",
        help="Path to the folder containing input INKML files",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output pickle file path"
    )
    parser.add_argument(
        "--use_los",
        action="store_true",
        help="Enable line-of-sight filtering in JSON reading",
    )
    args = parser.parse_args()

    # print args
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # Create the dataset.
    math_graph_ds = MathGraph()
    math_graph_ds.graphs = []
    math_graph_ds.create_data(
        args.input_jsons, args.input_lgs, args.input_inkmls, use_los=args.use_los
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # Save the graphs to a pickle file.
    pickle.dump(math_graph_ds.graphs, open(args.output, "wb"))
