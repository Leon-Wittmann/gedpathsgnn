import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import networkx as nx
import os
from torch_geometric.data import Data, InMemoryDataset

class DataEdgeAttr(Data):
    pass

def build_dataset_paths(method: str, dataset: str):
    """
    Dynamische Pfad-Erstellung für data.pt und edit-path-Datei.
    """
    base_dir = os.path.join("data", "Results", method, dataset)
    data_path = os.path.join(base_dir, "processed", "data.pt")
    path_file = os.path.join(base_dir, f"{dataset}_edit_paths_data.txt")
    return data_path, path_file

def parse_graph_step(line: str):
    parts = line.strip().split()
    start_graph = int(parts[0])
    step = int(parts[1])
    target_graph = int(parts[2])
    level = parts[3]  # 'EDGE' oder 'NODE'
    element_info = parts[4]
    operation_type = parts[5]

    # Knoten/Kanten IDs aufteilen
    if level == "EDGE":
        node1, node2 = map(int, element_info.split("--"))
        element_info = (node1, node2)
    else:
        element_info = int(element_info)

    return {
        "start_graph": start_graph,
        "step": step,
        "target_graph": target_graph,
        "level": level,
        "element_info": element_info,
        "operation_type": operation_type
    }

def load_graphs_from_pt(method: str, dataset: str):
    data_path, path_file = build_dataset_paths(method, dataset)

    torch.serialization.add_safe_globals([Data])
    raw = torch.load(data_path, map_location="cpu", weights_only=False)

    data, slices = raw[0], raw[1]

    class DummyDataset(InMemoryDataset):
        def __init__(self):
            super().__init__()

    dataset_obj = DummyDataset()
    dataset_obj.data = data
    dataset_obj.slices = slices

    structured_steps = []
    with open(path_file, "r") as f:
        for line in f:
            if line.strip():
                step_struct = parse_graph_step(line)
                structured_steps.append(step_struct)

    return dataset_obj, structured_steps


