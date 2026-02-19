from collections import defaultdict
from torch_geometric.datasets import TUDataset
from loadGraphs import load_graphs_from_pt
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from models import GCN, GAT, GIN
from plot import plot_and_save_curves
import os
import pandas as pd
import dataframe_image as dfi
import random

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_ids(dataset_name):
    dataset = TUDataset(root='./data', name=dataset_name)
    new_graphs = []
    for i, graph in enumerate(dataset):
        new_graph = Data(x = graph.x, edge_index = graph.edge_index, y = graph.y, id = i)
        new_graphs.append(new_graph)
        
    dataset.data, dataset.slices = dataset.collate(new_graphs)
    return dataset

def training(dataset_name, method, augmentation_method, radius=1, model_class = GIN, original=None, aug_dataset=None, num_aug=0, dropout=0):
    epoch = 300
    hidden_dim = 256
    batch_size = 128
    test_dataset  = original[int(0.8 * len(original)):]

    if(augmentation_method == "baseline"):
        train_dataset = original[:int(0.8 * len(original))]
    else:
        train_dataset = original[:int(0.8 * len(original))] + aug_dataset
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    num_node_features = original.num_node_features
    num_classes = original.num_classes

    print(num_node_features)
    print(num_classes)

    model = model_class(in_channels=num_node_features, hidden_channels=hidden_dim, num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, test_accs, train_accs = [], [], []

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

        return total_loss / len(loader), (all_preds == all_labels).float().mean().item()

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
        loss, train_acc = train(model, train_loader, optimizer)
        acc = test(model, test_loader)
        train_losses.append(loss)
        test_accs.append(acc)
        train_accs.append(train_acc)
        print("Epoch: ", epoch, " | Loss: ", loss, " | Accuracy Test: ", acc, " | Accuracy Training: ", train_acc)

    os.makedirs("plots/augmentation", exist_ok=True)
    if(augmentation_method != "relative_all"):
        plot_and_save_curves(train_losses, test_accs, out_dir="plots/augmentation", prefix=f"{dataset_name}_{model.__class__.__name__}_{method}_{augmentation_method}_{radius}")
        MODEL_PATH = f"models/augmentation/{dataset_name}_{model.__class__.__name__}_{method}_{augmentation_method}_{radius}.pt"
    else:
        plot_and_save_curves(train_losses, test_accs, out_dir="plots/augmentation", prefix=f"{dataset_name}_{model.__class__.__name__}_{method}_{augmentation_method}_{radius}_{dropout}")
        MODEL_PATH = f"models/augmentation/{dataset_name}_{model.__class__.__name__}_{method}_{augmentation_method}_{radius}_{dropout}.pt"
    os.makedirs("models/augmentation", exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "num_node_features": num_node_features,
        "num_classes": num_classes,
        "hidden_dim": hidden_dim,
    }, MODEL_PATH)

    print(f"Modell gespeichert unter {MODEL_PATH}")

    if(augmentation_method == "baseline"):
        augmentation_method = "Basis Datensatz"
    elif(augmentation_method == "absolute"):
        augmentation_method = f"Absolut k={radius}"
    elif(augmentation_method == "relative"):
        augmentation_method = f"Relativ k={radius}"
    elif(augmentation_method == "absolute_all"):
        augmentation_method = f"Absolut_all k={radius}"
    elif(augmentation_method == "relative_all"):
        augmentation_method = f"Relativ k={radius} dropout={dropout}"

    return{
        "Datensatz": dataset_name,
        "Model": model.__class__.__name__,
        "Methode": augmentation_method,
        "Trainings Loss": train_losses[-1],
        "Trainings Accuracy": train_accs[-1],
        "Test Accuracy": test_accs[-1],
        "Anzahl neuer Graphen": num_aug
    }

def graph_augmentation_relative_dropout(dataset_name, method, radius , original, dropout):
    graphs_paths, steps = load_graphs_from_pt(method=method, dataset=dataset_name)
    paths = []
    for i, graph in enumerate(graphs_paths):
        parts = graph["bgf_name"].split("_")
        d_name, start, target, step = parts
        start = int(start)
        target = int(target)
        step = int(step)
        if(i == 0):
            paths.append({
                "start_id": start,
                "target_id": target,
                "length": step,
                "graphs": [graph]
            })
        elif(step == 0):
            paths.append({
                "start_id": start,
                "target_id": target,
                "length": step,
                "graphs": [graph]
            })

        else:
            paths[len(paths)-1]["length"] = step
            paths[len(paths)-1]["graphs"].append(graph)

    print(f"{len(paths)} Pfade eingelesen.")

    test_dataset  = original[int(0.8 * len(original)):]
    test_dataset_ids = []
    for i, graph in enumerate(test_dataset):
        test_dataset_ids.append(graph.id)
    new_graphs = []
    for i, path in enumerate(paths):
        start_id = path["start_id"]
        target_id = path["target_id"]
        length = path["length"]
        graphs = path["graphs"]
        radius_abs = int(len(graphs) * radius)

        if(start_id not in test_dataset_ids):
            for i in range(1, radius_abs+1):
                if(random.random() > dropout):
                    graph = graphs[i]
                    new_graph = Data(x = graph.x, edge_index = graph.edge_index, y = torch.tensor([original[start_id].y.item()]), id=-1)
                    new_graph.x = new_graph.x[:, -original.num_node_features:]
                    new_graphs.append(new_graph)

        if(target_id not in test_dataset_ids):
            for i in range(len(graphs) - (radius_abs + 1), len(graphs) - 1):
                if(random.random() > dropout):
                    graph = graphs[i]
                    new_graph = Data(x = graph.x, edge_index = graph.edge_index, y = torch.tensor([original[target_id].y.item()]), id=-1)
                    new_graph.x = new_graph.x[:, -original.num_node_features:]
                    new_graphs.append(new_graph)

    graphs_paths.data, graphs_paths.slices = graphs_paths.collate(new_graphs)
    print("Datensatz erweitert mit ", len(new_graphs), " Graphen.")
    return graphs_paths, len(new_graphs)

def graph_augmentation_all(dataset_name, method, radius ,original):
    graphs_paths, steps = load_graphs_from_pt(method=method, dataset=dataset_name)
    paths = []
    for i, graph in enumerate(graphs_paths):
        parts = graph["bgf_name"].split("_")
        d_name, start, target, step = parts
        start = int(start)
        target = int(target)
        step = int(step)
        if(i == 0):
            paths.append({
                "start_id": start,
                "target_id": target,
                "length": step,
                "graphs": [graph]
            })
        elif(step == 0):
            paths.append({
                "start_id": start,
                "target_id": target,
                "length": step,
                "graphs": [graph]
            })

        else:
            paths[len(paths)-1]["length"] = step
            paths[len(paths)-1]["graphs"].append(graph)

    print(f"{len(paths)} Pfade eingelesen.")

    test_dataset  = original[int(0.8 * len(original)):]
    test_dataset_ids = []
    for i, graph in enumerate(test_dataset):
        test_dataset_ids.append(graph.id)
    new_graphs = []
    for i, path in enumerate(paths):
        start_id = path["start_id"]
        target_id = path["target_id"]
        length = path["length"]
        graphs = path["graphs"]

        if(start_id not in test_dataset_ids):
            for i in range(1, radius+1):
                graph = graphs[i]
                new_graph = Data(x = graph.x, edge_index = graph.edge_index, y = torch.tensor([original[start_id].y.item()]), id=-1)
                new_graph.x = new_graph.x[:, -original.num_node_features:]
                new_graphs.append(new_graph)

        if(target_id not in test_dataset_ids):
            for i in range(len(graphs) - (radius + 1), len(graphs) - 1):
                graph = graphs[i]
                new_graph = Data(x = graph.x, edge_index = graph.edge_index, y = torch.tensor([original[target_id].y.item()]), id=-1)
                new_graph.x = new_graph.x[:, -original.num_node_features:]
                new_graphs.append(new_graph)

    graphs_paths.data, graphs_paths.slices = graphs_paths.collate(new_graphs)
    print("Datensatz erweitert mit ", len(new_graphs), " Graphen.")
    return graphs_paths, len(new_graphs)


def graph_augmentation(dataset_name, method, radius, original):
    test_dataset  = original[int(0.8 * len(original)):]
    test_dataset_ids = []
    for i, graph in enumerate(test_dataset):
        test_dataset_ids.append(graph.id)
            
    graphs, steps = load_graphs_from_pt(method=method, dataset=dataset_name)
    new_graphs = []
    augmented = []
    for i, graph in enumerate(graphs):
        print(i)
        parts = graph["bgf_name"].split("_")
        d_name, start, target, step = parts
        start = int(start)
        target = int(target)
        step = int(step)
        if(step == radius and step != 0):
            if((start not in test_dataset_ids) and (start not in augmented)):
                new_graph = Data(x = graph.x, edge_index = graph.edge_index, y = torch.tensor([original[start].y.item()]), id=-1)
                new_graph.x = new_graph.x[:, -original.num_node_features:]
                new_graphs.append(new_graph)
                augmented.append(start)

    graphs.data, graphs.slices = graphs.collate(new_graphs)
    print("Datensatz erweitert mit ", len(graphs), " Graphen.")
    return graphs, len(new_graphs)

def graph_augmentation_relative(dataset_name, method, radius, original):
    test_dataset  = original[int(0.8 * len(original)):]
    test_dataset_ids = []
    for i, graph in enumerate(test_dataset):
        test_dataset_ids.append(graph.id)
    graphs, steps = load_graphs_from_pt(method=method, dataset=dataset_name)
    mappings = pd.read_csv(f"data\Results\Mappings\{method}\{dataset_name}\{dataset_name}_ged_mapping.csv")
    mappings.columns = (mappings.columns.str.strip().str.lower().str.replace(" ", "_"))
    x = 0
    count_aug_graphs = 0
    new_graphs = []
    augmented = []
    for row in mappings.itertuples(index=False):
        source = row.source_id
        target_id = row.target_id
        dist = row.approximated_distance
        arg_graph_id = int(dist*radius)
        for i in range(x, len(graphs)):
            graph = graphs[i]
            parts = graph["bgf_name"].split("_")
            d_name, start, target, step = parts
            start = int(start)
            target = int(target)
            step = int(step)
            if(start == source and target == target_id and step == arg_graph_id and step != 0):
                if start not in test_dataset_ids:
                    if start not in augmented:
                        count_aug_graphs += 1
                        augmented.append(start)
                        new_graph = Data(x = graph.x, edge_index = graph.edge_index, y = torch.tensor([original[start].y.item()]), id=-1)
                        new_graph.x = new_graph.x[:, -original.num_node_features:]
                        new_graphs.append(new_graph)
                        print("Gefunden.", count_aug_graphs, step)
                x = i
                break        

    graphs.data, graphs.slices = graphs.collate(new_graphs)             
    return graphs, len(new_graphs)         

def run_test():
    datasets = ["Mutagenicity", "DHFR", "NCI1", "NCI109", "IMDB-BINARY", "IMDB-MULTI"]
    method = "BIPARTITE"
    models = [GIN, GCN, GAT]
    for dataset_name in datasets:
        results = []
        original = set_ids(dataset_name=dataset_name).shuffle()
        dataset_5, len5 = graph_augmentation(dataset_name=dataset_name, method=method, radius=5, original=original)
        dataset_10, len10 = graph_augmentation(dataset_name=dataset_name, method=method, radius=10, original=original)
        dataset0_2, len0_2 = graph_augmentation_relative(dataset_name=dataset_name, method=method, radius=0.2, original=original)
        dataset0_4, len0_4 = graph_augmentation_relative(dataset_name=dataset_name, method=method, radius=0.4, original=original)

        for model_class in models:
            print("Teilweise fertig.")
            results.append(training(dataset_name=dataset_name, method="BIPARTITE", augmentation_method="baseline", model_class=model_class, original=original))
            results.append(training(dataset_name=dataset_name, method="BIPARTITE", augmentation_method="absolute", radius=5, model_class=model_class, original=original, aug_dataset=dataset_5, num_aug=len5))
            results.append(training(dataset_name=dataset_name, method="BIPARTITE", augmentation_method="absolute", radius=10, model_class=model_class, original=original, aug_dataset=dataset_10, num_aug=len10))
            results.append(training(dataset_name=dataset_name, method="BIPARTITE", augmentation_method="relative", radius=0.2, model_class=model_class, original=original, aug_dataset=dataset0_2, num_aug=len0_2))
            results.append(training(dataset_name=dataset_name, method="BIPARTITE", augmentation_method="relative", radius=0.4, model_class=model_class, original=original, aug_dataset=dataset0_4, num_aug=len0_4))

        df = pd.DataFrame(results)
        df = df.round(4)
        os.makedirs("plots/augmentation", exist_ok=True)
        dfi.export(df, f"plots/augmentation/gnn_augmentation_results_{dataset_name}.png")

def run_test_all():
    datasets = ["Mutagenicity", "DHFR", "NCI1", "NCI109", "IMDB-BINARY", "IMDB-MULTI"]
    method = "BIPARTITE"
    models = [GIN, GCN, GAT]
    for dataset_name in datasets:
        results = []
        original = set_ids(dataset_name=dataset_name).shuffle()
        dataset_1, len1 = graph_augmentation_all(dataset_name=dataset_name, method=method, radius=1, original=original)
        dataset_2, len2 = graph_augmentation_all(dataset_name=dataset_name, method=method, radius=2, original=original)
        dataset_3, len3 = graph_augmentation_all(dataset_name=dataset_name, method=method, radius=3, original=original)
        dataset_4, len4 = graph_augmentation_all(dataset_name=dataset_name, method=method, radius=4, original=original)
        dataset_5, len5 = graph_augmentation_all(dataset_name=dataset_name, method=method, radius=5, original=original)

        for model_class in models:
            print("Teilweise fertig.")
            results.append(training(dataset_name=dataset_name, method="BIPARTITE", augmentation_method="baseline", model_class=model_class, original=original))
            results.append(training(dataset_name=dataset_name, method="BIPARTITE", augmentation_method="absolute_all", radius=1, model_class=model_class, original=original, aug_dataset=dataset_1, num_aug=len1))
            results.append(training(dataset_name=dataset_name, method="BIPARTITE", augmentation_method="absolute_all", radius=2, model_class=model_class, original=original, aug_dataset=dataset_2, num_aug=len2))
            results.append(training(dataset_name=dataset_name, method="BIPARTITE", augmentation_method="absolute_all", radius=3, model_class=model_class, original=original, aug_dataset=dataset_3, num_aug=len3))
            results.append(training(dataset_name=dataset_name, method="BIPARTITE", augmentation_method="absolute_all", radius=4, model_class=model_class, original=original, aug_dataset=dataset_4, num_aug=len4))
            results.append(training(dataset_name=dataset_name, method="BIPARTITE", augmentation_method="absolute_all", radius=5, model_class=model_class, original=original, aug_dataset=dataset_5, num_aug=len5))
           

        df = pd.DataFrame(results)
        df = df.round(4)
        os.makedirs("plots/augmentation", exist_ok=True)
        dfi.export(df, f"plots/augmentation/gnn_augmentation_results_absolute_all_{dataset_name}.png")

def run_test_all_dropout():
    datasets = ["Mutagenicity", "DHFR", "NCI1", "NCI109", "IMDB-BINARY", "IMDB-MULTI"]
    method = "BIPARTITE"
    models = [GIN, GCN, GAT]
    for dataset_name in datasets:
        results = []
        original = set_ids(dataset_name=dataset_name).shuffle()
        dataset_1, len1 = graph_augmentation_relative_dropout(dataset_name=dataset_name, method=method, radius=0.1, original=original, dropout = 0.5)
        dataset_2, len2 = graph_augmentation_relative_dropout(dataset_name=dataset_name, method=method, radius=0.25, original=original, dropout = 0.8)
        dataset_3, len3 = graph_augmentation_relative_dropout(dataset_name=dataset_name, method=method, radius=0.5, original=original, dropout = 0.9)

        for model_class in models:
            print("Teilweise fertig.")
            results.append(training(dataset_name=dataset_name, method="BIPARTITE", augmentation_method="baseline", model_class=model_class, original=original))
            results.append(training(dataset_name=dataset_name, method="BIPARTITE", augmentation_method="relative_all", radius=0.1, model_class=model_class, original=original, aug_dataset=dataset_1, num_aug=len1, dropout = 0.5))
            results.append(training(dataset_name=dataset_name, method="BIPARTITE", augmentation_method="relative_all", radius=0.25, model_class=model_class, original=original, aug_dataset=dataset_2, num_aug=len2, dropout = 0.8))
            results.append(training(dataset_name=dataset_name, method="BIPARTITE", augmentation_method="relative_all", radius=0.5, model_class=model_class, original=original, aug_dataset=dataset_3, num_aug=len3, dropout = 0.9))
           

        df = pd.DataFrame(results)
        df = df.round(4)
        os.makedirs("plots/augmentation", exist_ok=True)
        dfi.export(df, f"plots/augmentation/gnn_augmentation_results_relative_all_dropout_{dataset_name}.png")

run_test_all()