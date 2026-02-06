from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.utils import to_networkx
import networkx as nx

def get_num_components(graph):
    adj = to_scipy_sparse_matrix(
        graph.edge_index,
        num_nodes= graph.num_nodes
    )
    n_components, labels = connected_components(adj, directed=False)
    return(int(n_components))

def get_num_k_circles(graph, lenC):

    G = to_networkx(graph, to_undirected=True)
    all_cycles = nx.cycle_basis(G)
    k_cycles = [c for c in all_cycles if len(c) == lenC]
    
    return len(k_cycles)