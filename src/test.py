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
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import connected_components
from utils import get_num_components, get_num_k_circles
from matplotlib.colors import PowerNorm

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
    
def average_first_flip(dataset_name, model_class, method, groups_start_target_label):
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
    plt.title("Durchschnittliche Anzahl an Schritten bis zum ersten Flip")

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value = df.values[i, j]
            if not np.isnan(value):
                plt.text(
                    j, i, f"{value:.2f}",
                    ha="center", va="center", color="black"
                )
            else:
                plt.text(
                    j, i, "None",
                    ha="center", va="center", color="black"
                )

    output_dir = "plots/average_first_flip"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_{model_class.__name__}_{method}.png", dpi=300)
    plt.close()

def create_flip_statistic(dataset_name, model_class, method, groups, flip_counts):
    rows = []

    for flips, eps in sorted(groups.items()):
        n = len(eps)

        def mean_std(values):
            mean = np.mean(values)
            if len(values) > 1:
                std = np.std(values, ddof=1)
            else:
                std = 0
            return f"{mean:.2f} ± {std:.2f}"


        row = {
            "number_of_flips": flips,
            "count": n,
            "nodes_start": mean_std([ep.number_of_nodes_start for ep in eps]),
            "edges_start": mean_std([ep.number_of_edges_start for ep in eps]),
            "nodes_target": mean_std([ep.number_of_nodes_target for ep in eps]),
            "edges_target": mean_std([ep.number_of_edges_target for ep in eps]),
            "Average GED": mean_std([ep.ged for ep in eps]),
        }

        rows.append(row)

    all_eps = [ep for eps_list in groups.values() for ep in eps_list]
    row_all = {
        "number_of_flips": "Total",
        "count": len(all_eps),
        "nodes_start": mean_std([ep.number_of_nodes_start for ep in all_eps]),
        "edges_start": mean_std([ep.number_of_edges_start for ep in all_eps]),
        "nodes_target": mean_std([ep.number_of_nodes_target for ep in all_eps]),
        "edges_target": mean_std([ep.number_of_edges_target for ep in all_eps]),
        "Average GED": mean_std([ep.ged for ep in all_eps]),
    }
    rows.append(row_all)

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

def load_trained_model(model_class, model_path):

    checkpoint = torch.load(model_path, map_location=DEVICE)

    model = model_class(
        checkpoint["num_node_features"],
        checkpoint["hidden_dim"],
        checkpoint["num_classes"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded model:")
    print("  node_features:", checkpoint["num_node_features"])
    print("  hidden_dim:", checkpoint["hidden_dim"])
    print("  num_classes:", checkpoint["num_classes"])

    return model, checkpoint

def create_heatmap(dataset_name, model_class, method, result_list):
    longest_path = max(len(ep.operations) for ep in result_list)
    most_flips = max(ep.number_of_flips for ep in result_list)
    print(longest_path)
    print(most_flips)
    heatmap0 = np.zeros((most_flips, longest_path))
    heatmap1 = np.zeros((most_flips, longest_path))
    heatmap2 = np.zeros((most_flips, longest_path))
    for i, ep in enumerate(result_list):
        if(ep.start_graph_label == 0):
            num_flips = 0
            for j, pred in enumerate(ep.prediction):
                if(j != 0):
                    if(ep.prediction[j-1] != pred):
                        heatmap0[num_flips, j-1] += 1
                        num_flips += 1
        elif(ep.start_graph_label == 1):
            num_flips = 0
            for j, pred in enumerate(ep.prediction):
                if(j != 0):
                    if(ep.prediction[j-1] != pred):
                        heatmap1[num_flips, j-1] += 1
                        num_flips += 1
        elif(ep.start_graph_label == 2):
            num_flips = 0
            for j, pred in enumerate(ep.prediction):
                if(j != 0):
                    if(ep.prediction[j-1] != pred):
                        heatmap2[num_flips, j-1] += 1
                        num_flips += 1
        else:
            print("Fehler Heatmap")

    heatmap0_plot = heatmap0[:20, :50]
    heatmap1_plot = heatmap1[:20, :50]
    heatmap2_plot = heatmap2[:20, :50]

    vmin = min(heatmap0_plot.min(), heatmap1_plot.min(), heatmap2_plot.min())
    vmax = max(heatmap0_plot.max(), heatmap1_plot.max(), heatmap2_plot.max())

    if np.all(heatmap2_plot == 0):
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 8), constrained_layout=True)

        im0 = axs[0].imshow(heatmap0_plot, cmap="Blues", interpolation="nearest", vmin=vmin, vmax=vmax)
        axs[0].set_title("Heatmap Class 0")           
        axs[0].set_xlabel("Step")
        axs[0].set_ylabel("Flip")
        
        im1 = axs[1].imshow(heatmap1_plot, cmap="Blues", interpolation="nearest", vmin=vmin, vmax=vmax)
        axs[1].set_title("Heatmap Class 1")
        axs[1].set_xlabel("Step")
        axs[1].set_ylabel("Flip")
    else:
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(5, 12), constrained_layout=True)

        im0 = axs[0].imshow(heatmap0_plot, cmap="Blues", interpolation="nearest", vmin=vmin, vmax=vmax)
        axs[0].set_title("Heatmap Class 0")           
        axs[0].set_xlabel("Step")
        axs[0].set_ylabel("Flip")
        
        im1 = axs[1].imshow(heatmap1_plot, cmap="Blues", interpolation="nearest", vmin=vmin, vmax=vmax)
        axs[1].set_title("Heatmap Class 1")
        axs[1].set_xlabel("Step")
        axs[1].set_ylabel("Flip")

        im2 = axs[2].imshow(heatmap2_plot, cmap="Blues", interpolation="nearest", vmin=vmin, vmax=vmax)
        axs[2].set_title("Heatmap Class 2")
        axs[2].set_xlabel("Step")
        axs[2].set_ylabel("Flip")

    cbar = fig.colorbar(im0, ax=axs, orientation="horizontal", fraction=0.05, pad=0.08)
    cbar.set_label("Anzahl Flips")

    os.makedirs("plots/heatmaps", exist_ok=True)
    plt.savefig(f"plots/heatmaps/heatmaps_class_0_1_{dataset_name}_{model_class.__name__}_{method}.png", dpi=300, bbox_inches='tight')
    plt.close()

def analyze_operations(dataset_name, model_class, method, result_list):
    OPS = ["NODE_INSERT", "NODE_DELETE", "NODE_RELABEL",
        "EDGE_INSERT", "EDGE_DELETE", "EDGE_RELABEL"]

    flip_counts = Counter({op: 0 for op in OPS})
    operation_counts = Counter({op: 0 for op in OPS})
    components_increase_counts = 0
    components_decrease_counts = 0
    components_increase_counts_total = 0
    components_decrease_counts_total = 0
    circles_created_counts = 0
    circles_deleated_counts = 0
    circles_created_counts_total = 0
    circles_deleated_counts_total = 0

    for path in result_list:
        pred = path.prediction
        ops = path.operations
        comps = path.num_components
        circs = path.num_circles


        for i, op in enumerate(ops, start=1):
            if pred[i-1] != pred[i]:
                flip_counts[op] += 1
                if(comps[i-1] > comps[i] and op == "EDGE_INSERT"):
                    components_decrease_counts += 1
                    components_decrease_counts_total += 1
                if(circs[i-1] > circs[i] and op == "EDGE_DELETE"):
                    circles_deleated_counts += 1
                    circles_deleated_counts_total += 1
                if(comps[i-1] < comps[i] and op == "EDGE_DELETE"):
                    components_increase_counts += 1
                    components_increase_counts_total += 1
                if(circs[i-1] > circs[i] and op == "EDGE_INSERT"):
                    circles_created_counts += 1
                    circles_created_counts_total += 1
            else:
                if(comps[i-1] > comps[i] and op == "EDGE_INSERT"):
                    components_decrease_counts_total += 1
                if(circs[i-1] > circs[i] and op == "EDGE_DELETE"):
                    circles_deleated_counts_total += 1
                if(comps[i-1] < comps[i] and op == "EDGE_DELETE"):
                    components_increase_counts_total += 1
                if(circs[i-1] > circs[i] and op == "EDGE_INSERT"):
                    circles_created_counts_total += 1

        for op in ops:
            operation_counts[op] += 1



    flip_rel = {op: flip_counts[op] / operation_counts[op] if operation_counts[op] > 0 else 0 for op in OPS}


    edge_insert_total = operation_counts["EDGE_INSERT"] if operation_counts["EDGE_INSERT"] > 0 else 1
    edge_delete_total = operation_counts["EDGE_DELETE"] if operation_counts["EDGE_DELETE"] > 0 else 1

    edge_insert_rel = {
        "components_decrease": components_decrease_counts / edge_insert_total,
        "circles_created": circles_created_counts / edge_insert_total
    }
    edge_delete_rel = {
        "components_increase": components_increase_counts / edge_delete_total,
        "circles_deleated": circles_deleated_counts / edge_delete_total
    }

    edge_insert_rel_total = {
        "components_decrease": components_decrease_counts_total / edge_insert_total,
        "circles_created": circles_created_counts_total / edge_insert_total
    }
    edge_delete_rel_total = {
        "components_increase": components_increase_counts_total / edge_delete_total,
        "circles_deleated": circles_deleated_counts_total / edge_delete_total
    }


    rest_edge_insert = flip_rel["EDGE_INSERT"] - sum(edge_insert_rel.values())
    rest_edge_delete = flip_rel["EDGE_DELETE"] - sum(edge_delete_rel.values())
    rest_edge_insert_total = (1-edge_insert_rel_total["components_decrease"] - edge_insert_rel_total["circles_created"])
    rest_edge_delete_total = (1-edge_delete_rel_total["components_increase"] - edge_delete_rel_total["circles_deleated"])

    fig, ax = plt.subplots(figsize=(14,6))


    for op in ["NODE_INSERT", "NODE_DELETE", "NODE_RELABEL", "EDGE_INSERT", "EDGE_DELETE", "EDGE_RELABEL", "EDGE_INSERT_TOTAL", "EDGE_DELETE_TOTAL"]:
        if op == "EDGE_INSERT":
            ax.bar(op, rest_edge_insert, color='blue', alpha=0.7, label="Flip Rate (rest)")
            ax.bar(op, edge_insert_rel["components_decrease"], bottom=rest_edge_insert, color='orange', label="Disconnected Parts Changed")
            ax.bar(op, edge_insert_rel["circles_created"], bottom=rest_edge_insert + edge_insert_rel["components_decrease"], color='green', label="Number of Circles changes (length=6)")
        elif op == "EDGE_DELETE":
            ax.bar(op, rest_edge_delete, color='blue', alpha=0.7)
            ax.bar(op, edge_delete_rel["components_increase"], bottom=rest_edge_delete, color='orange')
            ax.bar(op, edge_delete_rel["circles_deleated"], bottom=rest_edge_delete + edge_delete_rel["components_increase"], color='green')
        elif op == "EDGE_INSERT_TOTAL":
            ax.bar(op, rest_edge_insert_total, color='blue', alpha=0.7, label="Flip Rate (rest)")
            ax.bar(op, edge_insert_rel_total["components_decrease"], bottom=rest_edge_insert_total, color='orange', label="Disconnected Parts Changed")
            ax.bar(op, edge_insert_rel_total["circles_created"], bottom=rest_edge_insert_total + edge_insert_rel_total["components_decrease"], color='green')
        elif op == "EDGE_DELETE_TOTAL":
            ax.bar(op, rest_edge_delete_total, color='blue', alpha=0.7)
            ax.bar(op, edge_delete_rel_total["components_increase"], bottom=rest_edge_delete_total, color='orange')
            ax.bar(op, edge_delete_rel_total["circles_deleated"], bottom=rest_edge_delete_total + edge_delete_rel_total["components_increase"], color='green')
        else:
            ax.bar(op, flip_rel[op], color='blue', alpha=0.7)


    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.set_ylabel("Relative Häufigkeit")
    ax.set_title("Flip-Statistik und Graph-Eigenschaftsänderungen pro Operation")

    plt.tight_layout()
    save_dir = "plots/operation_stats"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}_{model_class.__name__}_{method}.png")
    plt.savefig(save_path)
    plt.close()

def analyze_intensity(dataset_name, model_class, method, result_list):
    total_flip_intensity = 0.0
    total_flips = 0

    for path in result_list:
        total_flip_intensity += path.sum_flip_intensity
        total_flips += path.number_of_flips
    
    if total_flips > 0:
        average_flip_intensity = total_flip_intensity / total_flips
    else:
        average_flip_intensity = 0.0

    output_dir = os.path.join("plots", "intensity")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "average_flip_intensity.csv")
    
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model_class", "method", "average_flip_intensity", "total_flips"])
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            "dataset": dataset_name,
            "model_class": model_class.__name__,
            "method": method,
            "average_flip_intensity": average_flip_intensity,
            "total_flips": total_flips
        })
    
    print(f"Saved/updated: {csv_path}")

def analyze_results(dataset_name, model_class, method, result_list):
    flip_counts = []

    groups = defaultdict(list)
    groups_start_target_label = defaultdict(list) 

    for i, ep in enumerate(result_list):    
        ep.calculate_sum_flip_intensity()
        ep.analyzeStartTargetGraphs()
        ep.readGED()
        ep.doFirstFlip()
        flip_counts.append(ep.number_of_flips)
        groups[ep.number_of_flips].append(ep)
        groups_start_target_label[(ep.start_graph_label, ep.target_graph_label)].append(ep.first_flip_relative)

    analyze_intensity(dataset_name=dataset_name, model_class=model_class, method=method, result_list=result_list)
    average_first_flip(dataset_name=dataset_name, model_class=model_class, method=method, groups_start_target_label=groups_start_target_label)
    create_flip_statistic(dataset_name=dataset_name, model_class=model_class, method=method, groups=groups, flip_counts=flip_counts)
    analyze_operations(dataset_name=dataset_name, model_class=model_class, method=method, result_list=result_list)
    create_heatmap(dataset_name=dataset_name, model_class=model_class, method=method, result_list=result_list)

def run_experiment(dataset_name, model_class, method):

    model_path = f"models/{dataset_name}_{model_class.__name__}.pt"
    model, ckpt = load_trained_model(model_class=model_class, model_path=model_path)
    model.eval()
    graphs, steps = load_graphs_from_pt(method=method, dataset=dataset_name)
    dataset = TUDataset(root="./data", name=dataset_name)
    count_ones = 0
    count_zeros = 0
    path_id = 0
    global_id = 0
    result_list = []
    result_list.append(EditPath(start_graph_id=graphs[0].edit_path_start.item(), target_graph_id=graphs[0].edit_path_end.item(), dataset_name=dataset_name, method=method, dataset=dataset))
    new_path = True
    loader = DataLoader(graphs, batch_size=128, shuffle=False)
    test=0
    test2=0
    for batch_data in loader:
        batch_data = batch_data.to(DEVICE)

        if isinstance(model, GIN):
            num_features_model = model.conv1.nn[0].in_features
        else:
            num_features_model = model.conv1.in_channels
        
        graph_features = batch_data.x[:, -num_features_model:]

        with torch.no_grad():
            out = model(graph_features, batch_data.edge_index, batch_data.batch)

        preds = out.argmax(dim=1).cpu().tolist()
        probs = torch.softmax(out, dim=1)

        for i, pred_class in enumerate(preds):
            graph = graphs[global_id]

            if global_id > 0:
                new_path = False

            if(graph.edit_path_start.item() != result_list[path_id].start_graph_id or graph.edit_path_end.item() != result_list[path_id].target_graph_id):
                print("Beginn eines neuen Pfades!")
                path_id += 1
                result_list.append(EditPath(start_graph_id=graph.edit_path_start.item(), target_graph_id=graph.edit_path_end.item(), dataset_name=dataset_name, method=method, dataset=dataset))
                new_path = True

            step = steps[global_id - path_id - 1]
            
            result_list[path_id].prediction.append(pred_class)
            result_list[path_id].logits.append(probs[i].detach().cpu().numpy())
            result_list[path_id].num_components.append(get_num_components(graph))
            result_list[path_id].num_circles.append(get_num_k_circles(graph, 6))
        
            if(new_path != True): 
                result_list[path_id].operations.append(f"{step['level']}_{step['operation_type']}")

            if pred_class == 1:
                count_ones += 1
            else:
                count_zeros += 1

            global_id += 1
            
        print(global_id, "/", len(graphs))

    count_correct_00 = 0
    count_correct_00_total = 0
    count_correct_11 = 0
    count_correct_11_total = 0
    count_correct_01 = 0
    count_correct_01_total = 0
    count_correct_10 = 0
    count_correct_10_total = 0
    for result in result_list:
        if(result.start_graph_label == 0 and result.target_graph_label == 0):
            count_correct_00_total += 1
            if(result.prediction[0] == result.start_graph_label and result.prediction[-1] == result.target_graph_label):
                count_correct_00 += 1
        elif(result.start_graph_label == 0 and result.target_graph_label == 1):
            count_correct_01_total += 1
            if(result.prediction[0] == result.start_graph_label and result.prediction[-1] == result.target_graph_label):
                count_correct_01 += 1
        elif(result.start_graph_label == 1 and result.target_graph_label == 0):
            count_correct_10_total += 1
            if(result.prediction[0] == result.start_graph_label and result.prediction[-1] == result.target_graph_label):
                count_correct_10 += 1
        elif(result.start_graph_label == 1 and result.target_graph_label == 1):
            count_correct_11_total += 1
            if(result.prediction[0] == result.start_graph_label and result.prediction[-1] == result.target_graph_label):
                count_correct_11 += 1

    print(count_correct_00, count_correct_00_total)
    print(count_correct_11, count_correct_11_total)
    print(count_correct_01, count_correct_01_total)
    print(count_correct_10, count_correct_10_total)
    print(test)
    print(test2)
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
    dataset, steps = load_graphs_from_pt(method=method, dataset=dataset_name)
    for i in range(10):
        data = dataset[i]
        adj = to_scipy_sparse_matrix(
            data.edge_index,
            num_nodes=data.num_nodes
        )
        n_components, labels = connected_components(adj, directed=False)
        dataset[i].num_components = int(n_components)
        print(dataset[i])
        print(dataset[i].edit_path_start.item(), dataset[i].edit_path_end.item(), dataset[i].edit_path_step.item())

def run_full_experiment():
    datasets = ["Mutagenicity", "DHFR", "NCI1", "NCI109", "IMDB-BINARY", "IMDB-MULTI"]
    models = [GIN, GCN, GAT]

    for dataset_name in datasets:
        for model_class in models:
            run_experiment(dataset_name=dataset_name, model_class=model_class, method="BIPARTITE")

def run_full_training():
    datasets = ["NCI1"]
    models = [GIN, GCN, GAT]

    results = []

    for dataset_name in datasets:
        for model_class in models:
            print(f"\n=== Training {model_class.__name__} on {dataset_name} ===")
            if(dataset_name == "IMDB-MULTI" or dataset_name == "IMDB-BINARY"):
                result = run_training(
                    dataset_name=dataset_name,
                    model_class=model_class,
                    hidden_dim=256,
                    batch_size=32,
                    epoch=300,
                    lr=0.001
                )
            else:
                result = run_training(
                    dataset_name=dataset_name,
                    model_class=model_class,
                    hidden_dim=256,
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

run_full_experiment()

