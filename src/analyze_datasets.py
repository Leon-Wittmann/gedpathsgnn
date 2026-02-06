from torch_geometric.datasets import TUDataset
from collections import Counter
import matplotlib.pyplot as plt
import os

def analyze_dataset(dataset_name):
    dataset = TUDataset(root="./data", name=dataset_name)
    classes = [graph.y.item() for graph in dataset]
    counts = Counter(classes)
    labels = list(counts.keys())
    sizes = list(counts.values())


    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f"Anteil Klasse in {dataset_name}")
    plt.axis("equal")
    output_dir = f"plots/analyze_graphs/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{dataset_name}_classes.png", dpi=300)
    plt.close()
    
    

datasets = ["Mutagenicity", "DHFR", "NCI1", "NCI109", "IMDB-BINARY", "IMDB-MULTI"]

for dataset in datasets:
    analyze_dataset(dataset_name=dataset)