"""GCN, GraphSAGE, and GAT in one file using plain PyTorch message passing."""

import argparse
import copy
import random
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    split: dict[str, torch.Tensor]
    num_classes: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthetic_graph() -> GraphData:
    """Tiny two-class graph for checking every model without an OGB download."""
    x = torch.tensor([
        [1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3],
        [0.3, 0.7], [0.2, 0.8], [0.1, 0.9], [0.0, 1.0],
    ])
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    undirected = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4), (3, 4),
    ]
    directed = undirected + [(dst, src) for src, dst in undirected]
    edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    split = {
        "train": torch.tensor([0, 1, 4, 5]),
        "valid": torch.tensor([2, 6]),
        "test": torch.tensor([3, 7]),
    }
    return GraphData(x, edge_index, labels, split, num_classes=2)


def load_ogb(name: str, root: str, max_nodes: int | None, seed: int) -> GraphData:
    """Load OGB data; no PyG neural-network layer is used."""
    from ogb.nodeproppred import PygNodePropPredDataset

    dataset = PygNodePropPredDataset(name=name, root=root)
    graph = dataset[0]
    x = graph.x.float()
    y = graph.y.view(-1).long()
    edge_index = graph.edge_index.long()

    # Add reverse directions so every graph connection can pass messages both ways.
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    split = {key: value.view(-1) for key, value in dataset.get_idx_split().items()}

    if max_nodes is not None and max_nodes < x.size(0):
        n = max_nodes
        keep = (edge_index[0] < n) & (edge_index[1] < n)
        x, y, edge_index = x[:n], y[:n], edge_index[:, keep]
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        train_end, valid_end = int(0.6 * n), int(0.8 * n)
        split = {
            "train": order[:train_end],
            "valid": order[train_end:valid_end],
            "test": order[valid_end:],
        }
        print("Using a random 60/20/20 split for the node subset.")

    return GraphData(x, edge_index, y, split, dataset.num_classes)


class GCNLayer(nn.Module):
    """Degree-normalized neighbor aggregation with self-loops."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        nodes = torch.arange(num_nodes, device=x.device)
        src = torch.cat((edge_index[0], nodes))
        dst = torch.cat((edge_index[1], nodes))

        degree = torch.bincount(dst, minlength=num_nodes).to(x.dtype).clamp_min(1)
        weight = degree[src].rsqrt() * degree[dst].rsqrt()
        transformed = self.linear(x)
        messages = transformed[src] * weight.unsqueeze(1)
        output = torch.zeros(
            num_nodes, transformed.size(1), dtype=x.dtype, device=x.device
        )
        output.index_add_(0, dst, messages)
        return output + self.bias


class SAGEConv(nn.Module):
    """Concatenate each node with the mean of its neighbors."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(2 * in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        neighbor_sum = torch.zeros_like(x)
        neighbor_sum.index_add_(0, dst, x[src])
        degree = torch.bincount(dst, minlength=x.size(0)).to(x.dtype).clamp_min(1)
        neighbor_mean = neighbor_sum / degree.unsqueeze(1)
        return self.linear(torch.cat((x, neighbor_mean), dim=1))


class GATLayer(nn.Module):
    """Multi-head graph attention with a destination-wise edge softmax."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.attention_dropout = attention_dropout
        self.linear = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.attention_src = nn.Parameter(torch.empty(heads, out_channels))
        self.attention_dst = nn.Parameter(torch.empty(heads, out_channels))
        output_size = heads * out_channels if concat else out_channels
        self.bias = nn.Parameter(torch.zeros(output_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attention_src)
        nn.init.xavier_uniform_(self.attention_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        nodes = torch.arange(num_nodes, device=x.device)
        src = torch.cat((edge_index[0], nodes))
        dst = torch.cat((edge_index[1], nodes))

        # Project every node separately for every attention head.
        hidden = self.linear(x).view(num_nodes, self.heads, self.out_channels)

        # One unnormalized attention score for every edge and head.
        src_score = (hidden[src] * self.attention_src).sum(dim=-1)
        dst_score = (hidden[dst] * self.attention_dst).sum(dim=-1)
        scores = F.leaky_relu(src_score + dst_score, negative_slope=0.2)

        # Stable softmax over all incoming edges of each destination node.
        index = dst.unsqueeze(1).expand(-1, self.heads)
        maximum = torch.full(
            (num_nodes, self.heads), -torch.inf, dtype=x.dtype, device=x.device
        )
        maximum.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
        exponentials = torch.exp(scores - maximum[dst])
        denominator = torch.zeros(
            num_nodes, self.heads, dtype=x.dtype, device=x.device
        )
        denominator.scatter_add_(0, index, exponentials)
        attention = exponentials / denominator[dst].clamp_min(1e-12)
        attention = F.dropout(
            attention, p=self.attention_dropout, training=self.training
        )

        messages = hidden[src] * attention.unsqueeze(-1)
        output = torch.zeros(
            num_nodes,
            self.heads,
            self.out_channels,
            dtype=x.dtype,
            device=x.device,
        )
        output.index_add_(0, dst, messages)
        if self.concat:
            output = output.reshape(num_nodes, self.heads * self.out_channels)
        else:
            output = output.mean(dim=1)
        return output + self.bias


class GCN(nn.Module):
    def __init__(self, inputs: int, hidden: int, classes: int, dropout: float) -> None:
        super().__init__()
        self.layer1 = GCNLayer(inputs, hidden)
        self.layer2 = GCNLayer(hidden, classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layer2(x, edge_index)


class GraphSAGE(nn.Module):
    def __init__(self, inputs: int, hidden: int, classes: int, dropout: float) -> None:
        super().__init__()
        self.layer1 = SAGEConv(inputs, hidden)
        self.layer2 = SAGEConv(hidden, classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layer2(x, edge_index)


class GAT(nn.Module):
    def __init__(
        self, inputs: int, hidden: int, classes: int, heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.layer1 = GATLayer(
            inputs, hidden, heads=heads, concat=True, attention_dropout=dropout
        )
        self.layer2 = GATLayer(
            hidden * heads, classes, heads=1, concat=False,
            attention_dropout=dropout,
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.layer1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layer2(x, edge_index)


def build_model(args: argparse.Namespace, inputs: int, classes: int) -> nn.Module:
    if args.model == "gcn":
        return GCN(inputs, args.hidden, classes, args.dropout)
    if args.model == "graphsage":
        return GraphSAGE(inputs, args.hidden, classes, args.dropout)
    if args.model == "gat":
        return GAT(inputs, args.hidden, classes, args.heads, args.dropout)
    raise ValueError(f"Unknown model: {args.model}")


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor, index: torch.Tensor) -> float:
    return (logits[index].argmax(dim=1) == labels[index]).float().mean().item()


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    data = synthetic_graph() if args.synthetic else load_ogb(
        args.dataset, args.data_root, args.max_nodes, args.seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    data.split = {key: value.to(device) for key, value in data.split.items()}

    model = build_model(args, data.x.size(1), data.num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_valid = -1.0
    best_test = 0.0
    best_state = None

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"model={args.model} device={device} parameters={parameters:,} "
        f"nodes={data.x.size(0):,} edges={data.edge_index.size(1):,}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(
            logits[data.split["train"]], data.y[data.split["train"]]
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            train_acc = accuracy(logits, data.y, data.split["train"])
            valid_acc = accuracy(logits, data.y, data.split["valid"])
            test_acc = accuracy(logits, data.y, data.split["test"])
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_test = test_acc
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} loss={loss.item():.4f} "
                f"train={train_acc:.4f} valid={valid_acc:.4f} test={test_acc:.4f}"
            )

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    torch.save(best_state, args.output)
    print(
        f"Best validation={best_valid:.4f}, corresponding test={best_test:.4f}; "
        f"checkpoint={args.output}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GCN, GraphSAGE, and GAT implemented with plain PyTorch"
    )
    parser.add_argument(
        "--model", choices=("gcn", "graphsage", "gat"), default="gcn"
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--dataset", default="ogbn-arxiv")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4, help="First-layer GAT heads")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output", default="best_model.pt")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
