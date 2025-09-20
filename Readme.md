# Link Prediction with Graph Neural Networks for Handwritten Mathematical Expression Recognition

This repository contains the official source code and experimental setup for the paper: **"Link prediction Graph Neural Networks for structure recognition of Handwritten Mathematical Expressions"**.

The project implements a framework for recognizing handwritten mathematical expressions (HMEs) by modeling them as symbol-level graphs and using a Graph Neural Network (GNN) to refine their structure. The core contribution is framing mathematical structure recognition as a link prediction problem on a symbol-level graph.

## Methodology Overview

Our approach shifts from complex stroke-level representations to more efficient **symbol-level graphs**, reducing structural complexity. The recognition pipeline involves three main stages:

1.  **Symbol and Relation Recognition**: A deep **Bidirectional Long Short-Term Memory (BLSTM)** network processes the online handwritten data to perform simultaneous symbol segmentation, symbol recognition, and initial spatial relation classification. This model uses global context to improve accuracy across all three tasks.

2.  **Symbol-Level Graph Construction**: The recognized symbols and relations are used to build a primitive graph. A parser based on a **2D Cocke-Younger-Kasami (CYK) algorithm** explores all possible spatial relationships among the symbols to create a comprehensive graph. This graph is then filtered using a **Line-of-Sight (LOS) algorithm** to remove redundant edges, creating a balance between coverage and complexity.

3.  **Link Prediction with EGAT**: An **Edge-featured Graph Attention Network (EGAT)** refines the constructed graph. The EGAT model performs a link prediction task, deciding which edges to keep or remove to produce the final **Symbol Layout Tree (SLT)**.

![Methodology Overview](https://raw.githubusercontent.com/ntcuong2103/math_online_egat/refs/heads/master/asset/fig1.svg)

## Features

*   **Graph Construction**: Converts handwritten math expressions from the CROHME dataset into symbol-level graph structures using a deep BLSTM, a 2D-CYK parser, and LOS filtering.
*   **Model**: Implements an **Edge Graph Attention Network (EGAT)** for link prediction to refine graph structure. The model is built with PyTorch and the Deep Graph Library (DGL) and leverages both node and edge features for its predictions.
*   **Training & Evaluation**: Uses PyTorch Lightning for streamlined training and includes comprehensive scripts for evaluation on the CROHME benchmark.
*   **Data Processing**: Provides utilities for parsing LG files, constructing graph data, and preparing data for training and testing.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/ntcuong2103/math_online_egat.git
    cd math_online_egat
    ```

2.  Install the required dependencies. The code is tested with Python 3.8+.
    ```bash
    pip install torch dgl pytorch-lightning wandb tqdm pandas numpy networkx
    ```

3.  Prepare the **CROHME dataset** (2014, 2016, 2019). Place the JSON, LG, and INKML files into the `data/` directory, following the expected subfolder structure (e.g., `data/Test2014_primitive_json`, `data/Crohme_all_LGs`, etc.).

## Usage

#### 1. Data Preparation

Generate the pickle files required for training and testing by running the provided shell script:
```bash
bash create_pickle_data.sh
```

#### 2. Training

Train the EGAT model for link prediction using the PyTorch Lightning trainer. WandB is used for logging.
```bash
python trainer_lt.py
```

#### 3. Evaluation

Evaluate the model's performance on link prediction and full expression recognition:
```bash
# Evaluate link prediction and graph structure
bash evaluate_link.sh
bash evaluate.sh
```

## File Structure

*   `data.py`: Handles data loading, graph construction, and PyTorch dataset creation.
*   `model.py`: Defines the EGAT model architecture and the PyTorch Lightning module.
*   `trainer_lt.py`: Main script for training the model.
*   `test_folder.py`: Script for batch evaluation on test sets.
*   `graph_evaluation_fn.py`: Contains functions for evaluating graph-level predictions.
*   `parse_lg.py`: Utilities for parsing and normalizing label graph (LG) files.
*   `utils/los.py`: Implementation of the Line-of-Sight (LOS) algorithm for edge filtering.
*   `HandsCTC.lbl`: Vocabulary file for symbols and relations.
*   `create_pickle_data.sh`, `evaluate_link.sh`, `evaluate.sh`: Utility scripts for data preparation and evaluation.

## Results

The model was evaluated on the CROHME 2014, 2016, and 2019 test sets. Our integrated `BLSTM-CYK & LOS` approach for graph generation followed by EGAT-based link prediction achieved competitive performance, outperforming other explicit graph-based methods like MST by a significant margin.

| Input Graph Method  | Test Set      | Expression Rate (%) | Structure Rate (%) |
| ------------------- | ------------- | ------------------- | ------------------ |
| **BLSTM-CYK & LOS** | CROHME 2014   | **37.59**           | **51.07**          |
| **BLSTM-CYK & LOS** | CROHME 2016   | **38.35**           | **48.60**          |
| **BLSTM-CYK & LOS** | CROHME 2019   | **38.86**           | **51.42**          |

An ablation study also confirmed that including a combined `NodeEdgeFeature` (concatenating source node, edge, and destination node features) significantly improves link prediction accuracy.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{Nguyen2024LinkPrediction,
  author    = {Cuong Tuan Nguyen and Ngoc Tuan Nguyen and Triet Hoang Minh Dao and Huy Minh Nhat Nguyen and Huy Truong Dinh},
  title     = {Link prediction Graph Neural Networks for structure recognition of Handwritten Mathematical Expressions},
  booktitle = {Proceedings of the International Conference on Document Analysis and Recognition (ICDAR)},
  year      = {2024}
}
```