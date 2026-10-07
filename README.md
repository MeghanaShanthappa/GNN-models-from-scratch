# GCN, GraphSAGE, and GAT from scratch

One Python program containing three node-classification models:

- **GCN:** degree-normalized neighbor sum
- **GraphSAGE:** mean neighbors, then concatenate with the node itself
- **GAT:** learned attention weight for every incoming edge

The message-passing layers use ordinary PyTorch operations such as `index_add_`,
`scatter_add_`, and `scatter_reduce_`. No `torch_geometric.nn` layer is used. OGB
and PyG are used only to load graph datasets.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Quick tests

```bash
python3 gnn_models.py --model gcn --synthetic --epochs 50
python3 gnn_models.py --model graphsage --synthetic --epochs 50
python3 gnn_models.py --model gat --synthetic --epochs 50
```

## OGB experiments

```bash
python3 gnn_models.py --model gcn --dataset ogbn-arxiv --epochs 100
python3 gnn_models.py --model graphsage --dataset ogbn-arxiv --epochs 100
python3 gnn_models.py --model gat --dataset ogbn-arxiv --epochs 100 --heads 4
```

For a quick CPU experiment, add:

```bash
--max-nodes 5000 --epochs 10 --cpu
```

When `--max-nodes` is used, the program makes a random 60/20/20 split. That run
is useful for testing but is not an official OGB benchmark result.

## Main difference

```text
GCN       : normalized sum of neighbor messages
GraphSAGE : Linear(self || mean(neighbors))
GAT       : weighted sum where attention learns each neighbor's importance
```
