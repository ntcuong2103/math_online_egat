from model import GNNTrainer
from glob import glob
import os
from data import create_graph_dgl
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="math-graph-attention/wpm6r99z/checkpoints/epoch=10-val_link_acc=0.9264.ckpt")
parser.add_argument("--input_jsons", type=str, default="data/Test2014_primitive_json")
parser.add_argument("--input_lgs", type=str, default="data/Crohme_all_LGs")
parser.add_argument("--input_inkmls", type=str, default="data/Test2014")
parser.add_argument("--mode", type=str, default="JSON_LOS")
parser.add_argument("--output", type=str, default="results.csv")
args = parser.parse_args()


model = GNNTrainer.load_from_checkpoint(
    args.checkpoint)

class GNNTest(GNNTrainer):
    def __init__(self, model):
        super(GNNTest, self).__init__()
        self.model = model
        self.model.eval()
        self.model.to(self.device)
    
    def test_folder(self, input_jsons, input_lgs, input_inkmls, mode = "JSON_LOS"):
        results = {}
        for fn in tqdm(glob(f"{input_jsons}/*.json")):
            fn_lg = os.path.join(
                input_lgs, os.path.basename(fn).replace(".json", ".lg")
            )
            fn_inkml = os.path.join(
                input_inkmls, os.path.basename(fn).replace(".json", ".inkml")
            )
            try:
                g = create_graph_dgl(fn, fn_lg, fn_inkml, mode)
                if g is not None and g.num_nodes() > 0 and g.num_edges() > 0:
                    loss, link_acc, accuracy = self.process_graph(g.to(self.device))
                    fn_id = os.path.basename(fn).replace(".json", "")
                    results[fn_id] = {
                        "link_acc": link_acc,
                        "accuracy": accuracy,
                    }
                else:
                    print(
                        f"Skipping graph: {fn}, num_nodes: {g.num_nodes()}, num_edges: {g.num_edges()}"
                    )
            except Exception as e:
                print(f"Error processing {fn}: {e}")
        return results

model_test = GNNTest(model.model)
results = model_test.test_folder(args.input_jsons, args.input_lgs, args.input_inkmls, args.mode)
# write to a csv file
with open(args.output, "w") as f:
    f.write("file_id,link_acc,accuracy\n")
    for fn, res in results.items():
        f.write(f"{fn},{res['link_acc']},{res['accuracy']}\n")



