"""Downsized, CPU-only, plain-script reconstruction of GNN_FINAL_notebook.ipynb's
core train/eval path (this repo's src/train.py is an empty stub -- the real
logic only exists in the Colab notebook, which has GPU-specific pip installs
and IPython `!magic` cells that don't run as a plain script). Same model
architecture, same features/labels, same F1 metric as the notebook and the
paper's slides -- just far fewer samples/epochs so it finishes in well under
a minute on CPU. Written by an assistant helping reproduce this repo's paper
claim locally; not part of the original commit history."""
import random
import heapq
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

N = 20
device = torch.device("cpu")

NUM_SAMPLES = 250   # notebook uses 1200 -- downsized for a quick CPU smoke test
NUM_EPOCHS = 8       # notebook uses 60


def random_grid(p_block=0.2):
    return [[1 if random.random() < p_block else 0 for _ in range(N)] for _ in range(N)]


def random_start_goal(grid):
    free_cells = [(i, j) for i in range(N) for j in range(N) if grid[i][j] == 0]
    if len(free_cells) < 2:
        return None, None
    start = random.choice(free_cells)
    goal = random.choice([c for c in free_cells if c != start])
    return start, goal


def neighbors(i, j):
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < N and 0 <= nj < N:
            yield ni, nj


def astar_shortest_path(grid, start, goal):
    def h(cell):
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    open_heap = [(h(start), 0, start)]
    came_from = {start: None}
    g_score = {start: 0}
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path
        i, j = cur
        for ni, nj in neighbors(i, j):
            if grid[ni][nj] == 1:
                continue
            nxt = (ni, nj)
            tentative_g = g + 1
            if nxt not in g_score or tentative_g < g_score[nxt]:
                g_score[nxt] = tentative_g
                came_from[nxt] = cur
                heapq.heappush(open_heap, (tentative_g + h(nxt), tentative_g, nxt))
    return None


def cell_to_index(i, j):
    return i * N + j


def grid_to_graph(grid, start, goal, path):
    num_nodes = N * N
    x = torch.zeros((num_nodes, 8), dtype=torch.float)
    y = torch.zeros(num_nodes, dtype=torch.float)
    path_set = set(path) if path is not None else set()
    start_i, start_j = start
    goal_i, goal_j = goal

    for i in range(N):
        for j in range(N):
            idx = cell_to_index(i, j)
            is_blocked = float(grid[i][j] == 1)
            is_start = float((i, j) == start)
            is_goal = float((i, j) == goal)
            x_norm = i / (N - 1)
            y_norm = j / (N - 1)
            man_goal = abs(i - goal_i) + abs(j - goal_j)
            man_start = abs(i - start_i) + abs(j - start_j)
            man_goal_norm = man_goal / (2 * (N - 1))
            man_start_norm = man_start / (2 * (N - 1))
            euclid = ((i - goal_i) ** 2 + (j - goal_j) ** 2) ** 0.5
            euclid_norm = euclid / ((2 * (N - 1)) ** 0.5)
            x[idx] = torch.tensor(
                [is_blocked, is_start, is_goal, x_norm, y_norm, man_goal_norm, man_start_norm, euclid_norm],
                dtype=torch.float,
            )
            if (i, j) in path_set:
                y[idx] = 1.0
            else:
                near = any((ni, nj) in path_set for ni, nj in neighbors(i, j))
                y[idx] = 0.3 if near else 0.0

    edge_index_list = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 1:
                continue
            idx = cell_to_index(i, j)
            for ni, nj in neighbors(i, j):
                if grid[ni][nj] == 1:
                    continue
                edge_index_list.append([idx, cell_to_index(ni, nj)])
    if not edge_index_list:
        return None
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, y=y)


def generate_dataset(num_samples, p_block=0.2):
    data_list = []
    while len(data_list) < num_samples:
        grid = random_grid(p_block=p_block)
        start, goal = random_start_goal(grid)
        if start is None:
            continue
        path = astar_shortest_path(grid, start, goal)
        if path is None:
            continue
        data = grid_to_graph(grid, start, goal, path)
        if data is not None:
            data_list.append(data)
    return data_list


class SAGEPathPredictor(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, dropout=0.2, num_layers=5):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_channels, hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x).squeeze(-1)


def compute_pos_weight(dataset):
    pos, neg = 0, 0
    for d in dataset:
        free_mask = d.x[:, 0] == 0
        y = d.y[free_mask]
        pos += (y == 1.0).sum().item()
        neg += (y == 0.0).sum().item()
    return torch.tensor([neg / (pos + 1e-8)], dtype=torch.float)


def _prf(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        free_mask = batch.x[:, 0] == 0
        loss = criterion(logits[free_mask], batch.y[free_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_y, all_pred = [], []
    for batch in loader:
        logits = model(batch.x, batch.edge_index)
        free_mask = batch.x[:, 0] == 0
        total_loss += criterion(logits[free_mask], batch.y[free_mask]).item()
        probs = torch.sigmoid(logits[free_mask])
        all_y.append(batch.y[free_mask])
        all_pred.append((probs > threshold).float())
    y_true = (torch.cat(all_y) >= 0.5).float()
    y_pred = torch.cat(all_pred)
    accuracy = (y_true == y_pred).float().mean().item()
    precision, recall, f1 = _prf(y_true, y_pred)
    return (total_loss / len(loader)), accuracy, precision, recall, f1


def main():
    print(f"Generating {NUM_SAMPLES} samples (downsized smoke test)...")
    dataset = generate_dataset(NUM_SAMPLES)
    random.shuffle(dataset)
    n = len(dataset)
    train_data = dataset[: int(0.7 * n)]
    val_data = dataset[int(0.7 * n) : int(0.85 * n)]
    test_data = dataset[int(0.85 * n) :]
    print(f"Split: train={len(train_data)} val={len(val_data)} test={len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model = SAGEPathPredictor(in_channels=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_weight = compute_pos_weight(train_data)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("pos_weight:", float(pos_weight.item()))

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc, va_prec, va_rec, va_f1 = evaluate(model, val_loader, criterion)
        print(
            f"Epoch {epoch:02d} | TrainLoss {tr_loss:.4f} | ValLoss {va_loss:.4f} | "
            f"ValAcc {va_acc:.4f} | ValScore {va_f1:.3f}"
        )

    best_t, best_f1 = 0.5, -1.0
    for k in range(21):
        t = k / 20
        _, _, _, _, f1 = evaluate(model, val_loader, criterion, threshold=t)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    print(f"Best threshold: {best_t:.2f} Best Val Score: {best_f1:.4f}")

    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion, threshold=best_t)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")
    print(f"Test Prec: {test_prec:.4f}")
    print(f"Test Rec:  {test_rec:.4f}")
    print(f"Test F1:   {test_f1:.4f}")


if __name__ == "__main__":
    main()
