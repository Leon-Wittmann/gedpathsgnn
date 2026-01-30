from collections import defaultdict
from torch_geometric.datasets import TUDataset
from loadGraphs import load_graphs_from_pt
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from models import GCN, GAT, GIN
from plot import plot_and_save_curves
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
def load_edit_paths(path):
    paths = []

    with open(path, "r") as f:
        for line in f:
            s_id, step, t_id, node_edge, comp, operation = line.strip().split()
            s_id, step, t_id = int(s_id), int(step), int(t_id)
            paths.append({"s_id":s_id, "step":step, "t_id":t_id, "node_edge":node_edge, "comp":comp, "operation":operation})

    paths = [step for step in paths if not step["step"] > 1]

    for i, step in enumerate(paths):
        print(step)

def graph_augmentation(dataset_name):

    dataset = TUDataset(root="data", name=dataset_name)
    paths = load_edit_paths(f"data/Results/BIPARTITE/{dataset_name}/{dataset_name}_edit_paths_data.txt")

graph_augmentation("DHFR")
"""

graphs, steps = load_graphs_from_pt("BIPARTITE", "DHFR")
dataset = TUDataset(root='./data', name="DHFR")

new_graphs = []
for i, graph in enumerate(graphs):
    parts = graph["bgf_name"].split("_")
    dataset_name, start, target, step = parts
    start = int(start)
    target = int(target)
    step = int(step)
    if(step == 1):
        new_graph = Data(x = graph.x, edge_index = graph.edge_index, y = torch.tensor([dataset[start].y.item()]))
        new_graph.x = new_graph.x[:, 1:]
        new_graphs.append(new_graph)
        
old_graphs = [dataset[i] for i in range(len(dataset))]
print(old_graphs[0])
print(new_graphs[0])
all_graphs = old_graphs + new_graphs
dataset.data, dataset.slices = dataset.collate(all_graphs)

print(len(dataset))

def training():
    epoch = 350
    train_dataset = dataset[:int(0.8 * len(dataset))]
    test_dataset  = dataset[int(0.8 * len(dataset)):]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=32)

    num_node_features = dataset.num_node_features
    num_classes = dataset.num_classes

    print(num_node_features)
    print(num_classes)

    model = GIN(in_channels=num_node_features, hidden_channels=64, num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_accs = [], []

    def train(model, loader, optimizer):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        for data in loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = torch.nn.functional.cross_entropy(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(data.y.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        pred_counts = torch.bincount(all_preds, minlength=num_classes)
        label_counts = torch.bincount(all_labels, minlength=num_classes)
        print("  Train label counts:", label_counts.tolist())
        print("  Train pred counts:", pred_counts.tolist())

        return total_loss / len(loader)

    def test(model, loader):
        model.eval()
        correct = 0
        all_preds = []
        all_labels = []
        for data in loader:
            data = data.to(DEVICE)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())

            all_preds.append(pred.cpu())
            all_labels.append(data.y.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        pred_counts = torch.bincount(all_preds, minlength=num_classes)
        label_counts = torch.bincount(all_labels, minlength=num_classes)
        print("  Test label counts:", label_counts.tolist())
        print("  Test pred counts:", pred_counts.tolist())

        return correct / len(loader.dataset)


    for epoch in range(1, epoch + 1):
        loss = train(model, train_loader, optimizer)
        acc = test(model, test_loader)
        train_losses.append(loss)
        test_accs.append(acc)
        print("Epoch: ", epoch, " | Loss: ", loss, " | Accuracy: ", acc)

    plot_and_save_curves(train_losses, test_accs, out_dir="plots", prefix=f"{dataset_name}_{model.__class__.__name__}")

    MODEL_PATH = f"models/augmentation/{dataset_name}_{model.__class__.__name__}.pt"
    os.makedirs("models/augmentation", exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "num_node_features": num_node_features,
        "num_classes": num_classes,
        "hidden_dim": 64,
    }, MODEL_PATH)

    print(f"Modell gespeichert unter {MODEL_PATH}")

training()