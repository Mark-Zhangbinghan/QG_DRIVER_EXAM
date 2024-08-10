import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号


# 建立结点类
class Vertex:
    def __init__(self, name, dot_weight, dot_place, near_dot):
        self.name = name
        self.dot_weight = dot_weight
        self.dot_place = dot_place
        self.near_dot = near_dot


# 根据信息找到结点的邻接点
def find_neighbors(num_list, neighbor_list):
    neighbors = []
    count = 1
    for num, neighbor in zip(num_list, neighbor_list):
        neighbor_ids = neighbor.split('、')  # 分割邻居结点序号
        for id in neighbor_ids:
            neighbors.append((num, int(id), str(count)))
            count += 1
    return neighbors


# 从excel表中获取结点信息G
def get_vertices(data_path):
    data = pd.read_excel(data_path)

    # 提取 X_Coordinate、Y_Coordinate 和 Name 列
    x_data = data['X_Coordinate']
    y_data = data['Y_Coordinate']
    nums = data['Num']
    names = data['Name']
    connect = data['Connect']
    weights = data['Weight']
    Edges = find_neighbors(nums, connect)

    # 创建字典
    Vertices = {num: (x, y) for num, x, y in zip(nums, x_data, y_data)}

    # 创建图
    G = nx.Graph()
    for node, pos in Vertices.items():
        G.add_node(node, pos=pos)
    for edge in Edges:
        G.add_edge(edge[0], edge[1], road=edge[2])
        pos1 = Vertices[edge[0]]
        pos2 = Vertices[edge[1]]
        distance = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        G.edges[edge[0], edge[1]]['length'] = distance

    # 设置节点属性
    nx.set_node_attributes(G, Vertices, 'pos')

    # 设置节点权重
    node_weights = {num: weight for num, weight in zip(nums, weights)}
    nx.set_node_attributes(G, node_weights, 'weight')

    return G, node_weights


# 建立结点数组
def create_vertices(data_path):
    G, node_weights = get_vertices(data_path)
    dot = []
    for node in G.nodes():
        near = []
        for edge in G.edges(node):
            if edge[0] == node:
                near.append(edge[1])
            elif edge[1] == node:
                near.append(edge[0])
        dot.append(Vertex(node, node_weights[node], G.nodes[node]['pos'], near))

    return G, dot


data_path = 'node_data.xlsx'
G, dot = create_vertices(data_path)  # G为集成的图像信息 dot是由结点类组成的数组
