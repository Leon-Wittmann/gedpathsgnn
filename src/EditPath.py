from torch_geometric.datasets import TUDataset
import csv
import os
import numpy as np

class EditPath:
    def __init__(self, start_graph_id, target_graph_id, dataset_name, method, dataset):
        self.start_graph_id = start_graph_id
        self.target_graph_id = target_graph_id
        self.dataset_name = dataset_name
        self.method = method
        self.dataset = dataset
        self.prediction = []
        self.logits = []
        self.length = 0
        self.number_of_flips = 0
        self.first_flip_relative = None
        self.number_of_nodes_start = 0
        self.number_of_nodes_target = 0
        self.number_of_edges_start = 0
        self.number_of_edges_target = 0
        self.ged = 0
        self.start_graph_label = dataset[start_graph_id].y.item()
        self.target_graph_label = dataset[target_graph_id].y.item()
        self.operations = []
        self.num_components = []
        self.num_circles = []
        self.sum_flip_intensity = 0

    """   
    def countFlips(self):
        for i, pr in enumerate(self.prediction):
            if(i > 0):
                if(self.prediction[i-1] != self.prediction[i]):
                    self.number_of_flips += 1
    """                

    def analyzeStartTargetGraphs(self):
        self.number_of_nodes_start = self.dataset[self.start_graph_id].num_nodes
        self.number_of_edges_start = self.dataset[self.start_graph_id].num_edges / 2
        self.number_of_nodes_target = self.dataset[self.target_graph_id].num_nodes
        self.number_of_edges_target = self.dataset[self.target_graph_id].num_edges / 2

    def readGED(self):
        path = os.path.join("data", "Results", "Mappings", self.method, self.dataset_name, f"{self.dataset_name}_ged_mapping.csv")
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)    
            for row in reader:
                start_id = int(row["source_id"])
                target_id = int(row[" target_id"])
                if (start_id == self.start_graph_id and target_id == self.target_graph_id):
                    self.ged = int(row[" approximated distance"])  
                    return  
                
    def doFirstFlip(self):
        flip_index = next(
            (i for i in range(1, len(self.prediction)) if self.prediction[i] != self.prediction[i-1]),
            None
        )
        if(flip_index != None ): 
            self.first_flip_relative = (flip_index+1)
        else:
            self.first_flip_relative = None


    def calculate_sum_flip_intensity(self):
        for i, pr in enumerate(self.prediction):
            if(i > 0):
                if(self.prediction[i-1] != self.prediction[i]):
                    self.number_of_flips += 1
                    self.sum_flip_intensity += np.abs(self.logits[i] - self.logits[i-1])
        
        