# Math Graph Attention

This repository provides code for training and evaluating graph neural network models (specifically EGAT-based) for mathematical expression recognition using graph representations of handwritten formulas. The project focuses on the CROHME dataset and leverages line-of-sight (LOS) algorithms to model relationships between strokes and symbols.

## Features

- **Graph Construction**: Converts handwritten math expressions into graph structures using JSON and LG (label graph) files, with optional LOS filtering.
- **Model**: Implements an Edge Graph Attention Network (EGAT) for node (symbol) and edge (relation) classification.
- **Training & Evaluation**: Uses PyTorch Lightning for training, with scripts for evaluation and result analysis.
- **Data Processing**: Utilities for parsing, normalizing, and visualizing graph data.

## Installation

1. Clone the repository.
2. Install dependencies (Python 3.8+, PyTorch, DGL, PyTorch Lightning, WandB, tqdm, pandas, numpy, networkx, etc.).
	 ```bash
	 pip install torch dgl pytorch-lightning wandb tqdm pandas numpy networkx
	 ```
3. Prepare the CROHME dataset and place the JSON, LG, and INKML files in the appropriate `data/` subfolders.

## Usage

### Data Preparation

- To generate pickle files for training/testing:
	```bash
	bash create_pickle_data.sh
	```

### Training

- Train the model using PyTorch Lightning:
	```bash
	python trainer_lt.py
	```

### Evaluation

- Evaluate link prediction and graph structure:
	```bash
	bash evaluate_link.sh
	bash evaluate.sh
	```

## File Structure

- `data.py`: Data loading, graph construction, and dataset utilities.
- `model.py`: EGAT model and PyTorch Lightning training module.
- `trainer_lt.py`: Training script with WandB logging.
- `test_folder.py`: Batch evaluation script for test sets.
- `graph_evaluation_fn.py`: Functions for evaluating graph predictions.
- `parse_lg.py`: Utilities for parsing and normalizing LG files.
- `utils/los.py`: Line-of-sight algorithm for graph edge construction.
- `HandsCTC.lbl`: Symbol and relation vocabulary.
- `create_pickle_data.sh`, `evaluate_link.sh`, `evaluate.sh`: Shell scripts for data preparation and evaluation.

## Data

- Place the CROHME dataset files in the `data/` directory as referenced in the scripts.
- The repository expects subfolders like `Test2014_primitive_json`, `Crohme_all_LGs`, etc.

## Citation

If you use this code, please cite the original authors and the CROHME dataset.

## License

MIT License (or specify if different).
