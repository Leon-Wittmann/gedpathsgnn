from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from collections import Counter
import matplotlib.pyplot as plt
import torch
import networkx as nx
import os

OUT_DIR = "plots/analyze_graphs"

def save_pie(counter, title, filename, dataset_name):
    if len(counter) == 0:
        return

    labels = list(counter.keys())
    sizes = list(counter.values())

    total = sum(sizes)
    sizes_pct = [s / total * 100 for s in sizes]

    plt.figure(figsize=(6, 6))
    plt.pie(
        sizes_pct,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title(title)
    plt.axis("equal")

    out_path = os.path.join(OUT_DIR, dataset_name, filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved {out_path}")

def analyze_dataset(dataset_name):
    dataset = TUDataset(root="./data", name=dataset_name)
    classes = [graph.y.item() for graph in dataset]
    counts = Counter(classes)
    labels = list(counts.keys())
    sizes = list(counts.values())

    node_counter = Counter()
    edge_counter = Counter()

    for data in dataset:
        G = to_networkx(data, to_undirected=True)
        if data.x is not None:
            node_labels = data.x.argmax(dim=1).tolist()
            node_counter.update(node_labels)

        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            if data.edge_attr.dim() > 1:
                edge_labels = data.edge_attr.argmax(dim=1).tolist()
            else:
                edge_labels = data.edge_attr.tolist()

            edge_counter.update(edge_labels)

    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f"Anteil Klasse in {dataset_name}")
    plt.axis("equal")
    output_dir = f"plots/analyze_graphs/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{dataset_name}_classes.png", dpi=300)
    plt.close()
    save_pie(node_counter, f"{dataset_name} Node Label Distribution", "node_label_distribution.png", dataset_name)
    save_pie(edge_counter, f"{dataset_name} Edge Label Distribution", "edge_label_distribution.png", dataset_name)

    

datasets = ["Mutagenicity", "DHFR", "NCI1", "NCI109", "IMDB-BINARY", "IMDB-MULTI"]

for dataset in datasets:
    analyze_dataset(dataset_name=dataset)