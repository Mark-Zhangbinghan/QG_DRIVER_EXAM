import matplotlib.pyplot as plt
import networkx as nx

Vertices = {1: (4, 4), 2: (18, 4), 3: (32, 4), 4: (4, 16), 5: (16, 14), 6: (28, 12), 7: (4, 22), 8: (34, 38), 9: (60, 34)}
Edges = [(1, 2), (1, 4), (2, 3), (2, 5), (3, 6), (4, 5), (4, 7), (5, 6), (5, 8), (6, 9), (7, 8), (8, 9)]


G = nx.Graph()
# 添加顶点和边
for edge in Edges:
    G.add_edge(*edge)
# 绘制图
nx.draw(G, Vertices, with_labels=True, node_color='lightblue', edge_color='gray')
plt.show()
