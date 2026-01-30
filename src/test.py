import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool 
from models import GCN, GAT, GIN
from plot import plot_and_save_curves
from loadGraphs import build_dataset_paths, load_graphs_from_pt
from torch_geometric.utils import to_networkx
from EditPath import EditPath
import matplotlib.pyplot as plt
import os
from collections import Counter
from collections import defaultdict
import pandas as pd
import dataframe_image as dfi
import csv
import numpy as np
import networkx as nx

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_training(dataset_name, model_class=GIN, hidden_dim=64, batch_size=32, epoch=350, lr=0.001):

    dataset = TUDataset(root='./data', name=dataset_name).shuffle()
    train_dataset = dataset[:int(0.8 * len(dataset))]
    test_dataset  = dataset[int(0.8 * len(dataset)):]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    num_node_features = dataset.num_node_features
    num_classes = dataset.num_classes

    print(num_node_features)
    print(num_classes)

    model = model_class(num_node_features, hidden_dim, num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    MODEL_PATH = f"models/{dataset_name}_{model.__class__.__name__}.pt"
    os.makedirs("models", exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "num_node_features": num_node_features,
        "num_classes": num_classes,
        "hidden_dim": hidden_dim,
    }, MODEL_PATH)

    print(f"Modell gespeichert unter {MODEL_PATH}")

    return{
        "dataset": dataset_name,
        "model": model.__class__.__name__,
        "train_loss": train_losses[-1],
        "test_acc": test_accs[-1]
    }


def load_trained_model(model_class, model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

    model = model_class(
        checkpoint["num_node_features"],
        checkpoint["hidden_dim"],
        checkpoint["num_classes"]
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint

def analyze_results(dataset_name, model_class, method, result_list):
    flip_counts = []

    groups = defaultdict(list)
    groups_start_target_label = defaultdict(list) 

    for i, ep in enumerate(result_list):    
        ep.countFlips()
        ep.analyzeStartTargetGraphs()
        ep.readGED()
        ep.doFirstFlip()
        flip_counts.append(ep.number_of_flips)
        groups[ep.number_of_flips].append(ep)
        groups_start_target_label[(ep.start_graph_label, ep.target_graph_label)].append(ep.first_flip_relative)

    #First Flip statistik

    table = defaultdict(dict)

    for(start_label, target_label), values in groups_start_target_label.items():
        clean_values = [v if v is not None else np.nan for v in values]
        table[start_label][target_label] = np.nanmean(clean_values)

    df = pd.DataFrame(table)
    df = df.sort_index().sort_index(axis=1)

    plt.figure(figsize=(8, 6))

    im = plt.imshow(df.values)

    plt.colorbar(im, label="Average First Flip")

    plt.xticks(
        ticks=np.arange(len(df.columns)),
        labels=df.columns
    )

    plt.yticks(
        ticks=np.arange(len(df.index)),
        labels=df.index
    )

    plt.xlabel("Target Graph Label")
    plt.ylabel("Start Graph Label")
    plt.title("Average First Flip per (Start, Target) Label")

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value = df.values[i, j]
            if not np.isnan(value):
                plt.text(
                    j, i, f"{value:.2f}",
                    ha="center", va="center", color="black"
                )

    output_dir = "plots/average_first_flip"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_{model_class.__name__}_{method}.png", dpi=300)
    plt.close()

    #Flip statistic speichern
    
    rows = []

    for flips, eps in sorted(groups.items()):
        n = len(eps)

        row = {
            "number_of_flips": flips,
            "count": n,
            "nodes_start_mean": round(
                sum(ep.number_of_nodes_start for ep in eps) / n, 2
            ),
            "edges_start_mean": round(
                sum(ep.number_of_edges_start for ep in eps) / n, 2
            ),
            "nodes_target_mean": round(
                sum(ep.number_of_nodes_target for ep in eps) / n, 2
            ),
            "edges_target_mean": round(
                sum(ep.number_of_edges_target for ep in eps) / n, 2
            ),
            "Average GED": round(
                sum(ep.ged for ep in eps) / n
            ),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    table_dir = "plots/flip_statistics"
    os.makedirs(table_dir, exist_ok=True)

    csv_path = os.path.join(
        table_dir,
        f"{dataset_name}_{model_class.__name__}_{method}.csv"
    )

    png_csv_path = os.path.join(
        table_dir,
        f"{dataset_name}_{model_class.__name__}_{method}.png"
    )

    df.to_csv(csv_path, index=False)
    dfi.export(df, png_csv_path)
    print(f"Table saved to {csv_path}")

    #flip_counts = [ep.number_of_flips for ep in result_list]
    flip_counter = Counter(flip_counts) 

    flip_values = sorted(flip_counter.keys())
    counts = [flip_counter[f] for f in flip_values]

    plt.figure(figsize=(8, 6))
    plt.bar(flip_values, counts, edgecolor='black')
    plt.xlabel("Anzahl Flips")
    plt.ylabel("Anzahl Pfade")
    plt.title(f"Flip Verteilung: {dataset_name} {model_class.__name__} {method}")
    plt.xticks(flip_values)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    save_dir = "plots/number_of_flips"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}_{model_class.__name__}_{method}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Histogram saved to {save_path}")

def run_experiment(dataset_name, model_class, method):

    model_path = f"models/{dataset_name}_{model_class.__name__}.pt"
    model, ckpt = load_trained_model(model_class=model_class, model_path=model_path)
    graphs, steps = load_graphs_from_pt(method=method, dataset=dataset_name)
    dataset = TUDataset(root="./data", name=dataset_name)
    count_ones = 0
    count_zeros = 0
    path_id = 0
    result_list = []
    result_list.append(EditPath(start_graph_id=steps[0]['start_graph'], target_graph_id=steps[0]['target_graph'], dataset_name=dataset_name, method=method, dataset=dataset))
    new_path = True
    flip_operation = []
    previous_pred_class = None

    for i, st in enumerate(steps):  

        graph = graphs[i + path_id + 1] #bgf beinhaltet target graph, .txt anweisungen nicht
        graph = graph.to(DEVICE)

        if isinstance(model, GIN):
            num_features_model = model.conv1.nn[0].in_features
        else:
            num_features_model = model.conv1.in_channels
        graph_features = graph.x[:, -num_features_model:].float()
        batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=DEVICE)

        new_path = False
        if(steps[i]['start_graph'] != result_list[path_id].start_graph_id or steps[i]['target_graph'] != result_list[path_id].target_graph_id ):
            print("Beginn eines neuen Pfades!")
            path_id += 1
            result_list.append(EditPath(start_graph_id=steps[i]['start_graph'], target_graph_id=steps[i]['target_graph'], dataset_name=dataset_name, method=method, dataset=dataset))
            new_path = True


        with torch.no_grad():
            out = model(graph_features, graph.edge_index.to(DEVICE), batch)
            
        
        pred_class = out.argmax(dim=1).item()

        if(new_path == False and previous_pred_class != None):
            if(pred_class != previous_pred_class):
                flip_operation.append((steps[i]['level'] ,steps[i]['operation_type']))

        previous_pred_class = pred_class
        result_list[path_id].prediction.append(pred_class)
        print(steps[i], pred_class)

        if pred_class == 1:
            count_ones += 1
        else:
            count_zeros += 1
    
    OPERATION_ORDER = [
        ("NODE", "INSERT"),
        ("NODE", "DELETE"),
        ("NODE", "RELABEL"),
        ("EDGE", "INSERT"),
        ("EDGE", "DELETE"),
        ("EDGE", "RELABEL"),
    ]

    counter = Counter(flip_operation)

    labels = []
    counts = []

    for op in OPERATION_ORDER:
        labels.append(f"{op[0]}_{op[1]}")
        counts.append(counter.get(op, 0))

    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Anzahl")
    plt.xlabel("Flip Operation")
    plt.title("Kritische Operationen")

    output_dir = "plots/operations"
    os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_{model_class.__name__}_{method}.png", dpi=300)

    analyze_results(dataset_name=dataset_name, model_class=model_class, method=method, result_list=result_list)

def save_dataframe_as_image(df, filename="gnn_results.png", dpi=300):
    fig, ax = plt.subplots(figsize=(len(df.columns) * 2, len(df) * 0.6 + 1))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close()

def test(dataset_name, method):
    dataset = TUDataset(root="./data", name=dataset_name)
    ep = EditPath(start_graph_id=0, target_graph_id=10, dataset_name=dataset_name, method=method, dataset=dataset)
    print(dataset[0].y.item())

#test(dataset_name="DHFR", method="BIPARTITE")    

run_experiment(dataset_name="Mutagenicity", model_class=GIN, method="BIPARTITE")
run_experiment(dataset_name="Mutagenicity", model_class=GCN, method="BIPARTITE")
run_experiment(dataset_name="Mutagenicity", model_class=GAT, method="BIPARTITE")

run_experiment(dataset_name="DHFR", model_class=GIN, method="BIPARTITE")
run_experiment(dataset_name="DHFR", model_class=GCN, method="BIPARTITE")
run_experiment(dataset_name="DHFR", model_class=GAT, method="BIPARTITE")

run_experiment(dataset_name="NCI1", model_class=GIN, method="BIPARTITE")
run_experiment(dataset_name="NCI1", model_class=GCN, method="BIPARTITE")
run_experiment(dataset_name="NCI1", model_class=GAT, method="BIPARTITE")

run_experiment(dataset_name="NCI109", model_class=GIN, method="BIPARTITE")
run_experiment(dataset_name="NCI109", model_class=GCN, method="BIPARTITE")
run_experiment(dataset_name="NCI109", model_class=GAT, method="BIPARTITE")

#run_training(dataset_name="IMDB-MULTI", model_class=GIN)
"""
datasets = ["Mutagenicity", "DHFR", "NCI1", "NCI109", "IMDB-BINARY", "IMDB-MULTI"]
models = [GIN, GCN, GAT]

#datasets = ["DHFR"]
#models = [GIN]

results = []

for dataset_name in datasets:
    for model_class in models:
        print(f"\n=== Training {model_class.__name__} on {dataset_name} ===")
        if(dataset_name == "IMDB-MULTI" or dataset_name == "IMDB-BINARY"):
            result = run_training(
                dataset_name=dataset_name,
                model_class=model_class,
                hidden_dim=32,
                batch_size=32,
                epoch=300,
                lr=0.001
            )
        else:
            result = run_training(
                dataset_name=dataset_name,
                model_class=model_class,
                hidden_dim=64,
                batch_size=32,
                epoch=300,
                lr=0.001
            )
        results.append(result)

df = pd.DataFrame(results)
df.to_csv("gnn_training_results.csv", index=False)
df_pretty = df.copy()
df_pretty["train_loss"] = df_pretty["train_loss"].round(4)
df_pretty["test_acc"] = (df_pretty["test_acc"] ).round(4)
df_no_index = df_pretty.reset_index(drop=True)
dfi.export(df_pretty, "gnn_training_results.png")
"""
