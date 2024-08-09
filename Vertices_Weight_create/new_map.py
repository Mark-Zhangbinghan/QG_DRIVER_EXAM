import networkx as nx
import numpy as np

# 初始化图和其他相关参数
G = nx.Graph()
Vertices = {'A': (4, 4), 'B': (18, 4), 'C': (32, 4), 'D': (4, 16), 'E': (16, 14), 'F': (28, 12), 'G': (4, 22), 'H': (34, 38), 'I': (60, 34)}
Edges = [('A', 'B', 'Road1'), ('A', 'D', 'Road2'), ('B', 'C', 'Road3'), ('B', 'E', 'Road4'), ('C', 'F', 'Road5'),
         ('D', 'E', 'Road6'), ('D', 'G', 'Road7'), ('E', 'F', 'Road8'), ('E', 'H', 'Road9'),
         ('F', 'I', 'Road10'), ('G', 'H', 'Road11'), ('H', 'I', 'Road12')]

node_weights = {'A': 10, 'B': 12, 'C': 5, 'D': 8, 'E': 6, 'F': 4, 'G': 11, 'H': 9, 'I': 10}

for node, pos in Vertices.items():
    G.add_node(node, pos=pos, weight=node_weights[node])

for edge in Edges:
    G.add_edge(edge[0], edge[1], road=edge[2])
    pos1 = Vertices[edge[0]]
    pos2 = Vertices[edge[1]]
    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    G.edges[edge[0], edge[1]]['length'] = distance