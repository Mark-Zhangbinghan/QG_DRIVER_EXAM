import matplotlib.pyplot as plt
import networkx as nx

Vertices = {'A': (4, 4), 'B': (18, 4), 'C': (32, 4), 'D': (4, 16), 'E': (16, 14), 'F': (28, 12), 'G': (4, 22), 'H': (34, 38), 'I': (60, 34)}
Edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('B', 'E'), ('C', 'F'), ('D', 'E'), ('D', 'G'), ('E', 'F'), ('E', 'H'), ('F', 'I'), ('G', 'H'), ('H', 'I')]


G = nx.Graph()
# 添加顶点和边
for edge in Edges:
    G.add_edge(*edge)
# 绘制图
nx.draw(G, Vertices, with_labels=True, node_color='lightblue', edge_color='gray')
plt.show()
