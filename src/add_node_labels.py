import os
import networkx as nx
import random

def read_tudataset(folder, prefix):

    edges_file = os.path.join(folder, f'{prefix}_A.txt')
    edges = []
    with open(edges_file, 'r') as f:
        for line in f:
            u, v = map(int, line.strip().split(','))
            edges.append((u, v))


    indicator_file = os.path.join(folder, f'{prefix}_graph_indicator.txt')
    graph_indicator = []
    with open(indicator_file, 'r') as f:
        for line in f:
            graph_indicator.append(int(line.strip()))


    node_labels_file = os.path.join(folder, f'{prefix}_node_labels.txt')
    node_labels = []
    if os.path.exists(node_labels_file):
        with open(node_labels_file, 'r') as f:
            for line in f:
                node_labels.append(int(line.strip()))
    else:
        node_labels = [0] * len(graph_indicator)


    graph_labels_file = os.path.join(folder, f'{prefix}_graph_labels.txt')
    graph_labels = []
    with open(graph_labels_file, 'r') as f:
        for line in f:
            graph_labels.append(int(line.strip()))


    graphs = {}
    for idx, graph_id in enumerate(graph_indicator):
        if graph_id not in graphs:
            graphs[graph_id] = nx.Graph()
        graphs[graph_id].add_node(idx + 1, label=node_labels[idx])

    for u, v in edges:

        graph_id = graph_indicator[u - 1]  
        graphs[graph_id].add_edge(u, v)

    return graphs, graph_labels

def set_all_node_labels(graphs, min_label=1, max_label=10):
    for G in graphs.values():

        labels = {n: random.randint(min_label, max_label) for n in G.nodes()}
        nx.set_node_attributes(G, labels, 'label')

def write_tudataset(graphs, graph_labels, output_folder, prefix):
    os.makedirs(output_folder, exist_ok=True)


    A_lines = []
    graph_indicator_lines = []
    node_label_lines = []
    node_offset = 0

    for graph_id, G in sorted(graphs.items()):
        mapping = {n: n + node_offset for n in G.nodes()} 
        for u, v in G.edges():
            A_lines.append(f"{mapping[u]},{mapping[v]}")
        for n in G.nodes():
            graph_indicator_lines.append(str(graph_id))
            node_label_lines.append(str(G.nodes[n]['label']))
        node_offset += G.number_of_nodes()


    def write_list(lines, path):
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

    write_list(A_lines, os.path.join(output_folder, f'{prefix}_A.txt'))
    write_list(graph_indicator_lines, os.path.join(output_folder, f'{prefix}_graph_indicator.txt'))
    write_list(node_label_lines, os.path.join(output_folder, f'{prefix}_node_labels.txt'))
    write_list([str(l) for l in graph_labels], os.path.join(output_folder, f'{prefix}_graph_labels.txt'))

input_folder = 'data/IMDB-MULTI'
output_folder = 'data/IMDB-MULTI_modified'
prefix = 'IMDB-MULTI'

graphs, graph_labels = read_tudataset(input_folder, prefix)
set_all_node_labels(graphs, min_label=1, max_label=1)
write_tudataset(graphs, graph_labels, output_folder, prefix)

print("Dataset erfolgreich modifiziert und gespeichert!")