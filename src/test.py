import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool 
from models import GCN, GAT, GIN
from plot import plot_and_save_curves
from editpaths import build_dataset_paths, load_graphs_from_pt
from EditPath import EditPath
import matplotlib.pyplot as plt
import os
from collections import Counter
from collections import defaultdict
import pandas as pd
import dataframe_image as dfi

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

def run_experiment(dataset_name, model_class, method):

    model_path = f"models/{dataset_name}_{model_class.__name__}.pt"
    model, ckpt = load_trained_model(model_class=model_class, model_path=model_path)
    graphs, steps = load_graphs_from_pt(method=method, dataset=dataset_name)
    count_ones = 0
    count_zeros = 0
    path_id = 0
    result_list = []
    result_list.append(EditPath(start_graph_id=steps[0]['start_graph'], target_graph_id=steps[0]['target_graph'], dataset_name=dataset_name))

    for i, st in enumerate(steps):  

        graph = graphs[i + path_id] #bgf beinhaltet target graph, .txt anweisungen nicht
        graph = graph.to(DEVICE)

        if isinstance(model, GIN):
            num_features_model = model.conv1.nn[0].in_features
        else:
            num_features_model = model.conv1.in_channels
        graph_features = graph.x[:, -num_features_model:].float()
        batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=DEVICE)

        if(steps[i]['start_graph'] != result_list[path_id].start_graph_id or steps[i]['target_graph'] != result_list[path_id].target_graph_id ):
            print("Beginn eines neuen Pfades!")
            path_id += 1
            result_list.append(EditPath(start_graph_id=steps[i]['start_graph'], target_graph_id=steps[i]['target_graph'], dataset_name=dataset_name))

        with torch.no_grad():
            out = model(graph_features, graph.edge_index.to(DEVICE), batch)
            
        
        pred_class = out.argmax(dim=1).item()
        result_list[path_id].prediction.append(pred_class)
        print(steps[i], pred_class)

        if pred_class == 1:
            count_ones += 1
        else:
            count_zeros += 1

    print(count_ones)
    print(count_zeros)
    
    flip_counts = []
    number_of_nodes_start_sum = 0
    number_of_edges_start_sum = 0
    number_of_nodes_target_sum = 0
    number_of_edges_target_sum = 0
    groups = defaultdict(list)

    for i, ep in enumerate(result_list):
        ep.countFlips()
        ep.analyzeStartTargetGraphs()
        flip_counts.append(ep.number_of_flips)
        groups[ep.number_of_flips].append(ep)

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
    plt.xlabel("Number of flips")
    plt.ylabel("Number of EditPaths")
    plt.title(f"Flip distribution: {dataset_name} {model_class.__name__} {method}")
    plt.xticks(flip_values)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    save_dir = "plots/number_of_flips"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}_{model_class.__name__}_{method}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Histogram saved to {save_path}")

    return 0



run_experiment(dataset_name="DHFR", model_class=GIN, method="BIPARTITE")
run_experiment(dataset_name="DHFR", model_class=GCN, method="BIPARTITE")
run_experiment(dataset_name="DHFR", model_class=GAT, method="BIPARTITE")

#run_training(dataset_name='IMDB-BINARY', model_class=GIN)
#run_training(dataset_name='IMDB-BINARY', model_class=GAT)
#run_training(dataset_name='IMDB-BINARY', model_class=GCN)

