import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool

"""
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, 32)
        self.lin2 = nn.Linear(32, num_classes)
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        x = F.leaky_relu(x)
        x = self.lin2(x)
        return x
"""
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, hidden_channels//2)
        self.lin2 = nn.Linear(hidden_channels//2, hidden_channels//2)
        self.lin3 = nn.Linear(hidden_channels//2, num_classes)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)
        x = F.leaky_relu(x)
        #x = self.dropout(x)

        x = self.lin2(x)
        x = F.leaky_relu(x)
        #x = self.dropout(x)

        x = self.lin3(x)

        return x



"""
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.lin = nn.Linear(hidden_channels, 32)
        self.lin2 = nn.Linear(32, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        x = F.leaky_relu(x)
        x = self.lin2(x)
        return x
"""
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
        )
        self.conv2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
        )

        self.lin = nn.Linear(hidden_channels, hidden_channels//2)
        self.lin2 = nn.Linear(hidden_channels//2, hidden_channels//2)
        self.lin3 = nn.Linear(hidden_channels//2, num_classes)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)
        x = F.leaky_relu(x)
        #x = self.dropout(x)

        x = self.lin2(x)
        x = F.leaky_relu(x)
        #x = self.dropout(x)

        x = self.lin3(x)

        return x


"""
class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        nn2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.lin = nn.Linear(hidden_channels, 32)
        self.lin2 = nn.Linear(32, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        x = F.leaky_relu(x)
        x = self.lin2(x)
        return x
"""

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super().__init__()
        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        nn2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)

        self.lin = nn.Linear(hidden_channels, hidden_channels//2)
        self.lin2 = nn.Linear(hidden_channels//2, hidden_channels//2)
        self.lin3 = nn.Linear(hidden_channels//2, num_classes)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)
        x = F.leaky_relu(x)
        #x = self.dropout(x)

        x = self.lin2(x)
        x = F.leaky_relu(x)
        #x = self.dropout(x)

        x=self.lin3(x)

        return x
