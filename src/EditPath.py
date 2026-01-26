from torch_geometric.datasets import TUDataset

class EditPath:
    def __init__(self, start_graph_id, target_graph_id, dataset_name, prediction=None, length=None):
        self.start_graph_id = start_graph_id
        self.target_graph_id = target_graph_id
        self.dataset_name = dataset_name
        self.prediction = []
        self.length = 0
        self.number_of_flips = 0
        self.first_flip_relative = None
        self.number_of_nodes_start = 0
        self.number_of_edges_start = 0
        self.number_of_nodes_target = 0
        self.number_of_nodes_target = 0
        
    def countFlips(self):
        for i, pr in enumerate(self.prediction):
            if(i > 0):
                if(self.prediction[i-1] != self.prediction[i]):
                    self.number_of_flips += 1

    def analyzeStartTargetGraphs(self):
        dataset = TUDataset(root='./data', name=self.dataset_name)
        self.number_of_nodes_start = dataset[self.start_graph_id].num_nodes
        self.number_of_edges_start = dataset[self.start_graph_id].num_edges / 2
        self.number_of_nodes_target = dataset[self.target_graph_id].num_nodes
        self.number_of_edges_target = dataset[self.target_graph_id].num_edges / 2