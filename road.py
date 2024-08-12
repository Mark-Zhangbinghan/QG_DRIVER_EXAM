import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from Vertices_Weight_create.create_Vertices import create_vertices  # 引入第二段代码

def initialize_graph_from_dot(dot):
    """初始化图"""
    G = nx.Graph()
    for vertex in dot:
        G.add_node(vertex.name, pos=vertex.dot_place, cars=vertex.dot_weight)
    for vertex in dot:
        for neighbor in vertex.near_dot:
            pos1 = vertex.dot_place
            pos2 = dot[neighbor - 1].dot_place
            length = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos2[1] - pos2[1]) ** 2)
            G.add_edge(vertex.name, neighbor, length=length)
    return G

def compute_centrality(G):
    """计算 PageRank 和中介中心性"""
    pagerank_values = nx.pagerank(G, weight='length')
    betweenness_centrality_values = nx.betweenness_centrality(G, weight='length')
    nx.set_node_attributes(G, pagerank_values, 'pagerank')
    nx.set_node_attributes(G, betweenness_centrality_values, 'betweenness')
    return pagerank_values, betweenness_centrality_values

def attract_rank(G, alpha=0.5, beta=0.5):
    """计算 AttractRank 值"""
    attractiveness = {}
    for node in G.nodes:
        pagerank_score = G.nodes[node]['pagerank']
        betweenness_score = G.nodes[node]['betweenness']
        attract_rank_score = alpha * pagerank_score + beta * betweenness_score
        attractiveness[node] = attract_rank_score
    nx.set_node_attributes(G, attractiveness, 'attract_rank')
    return attractiveness

def heuristic(node, end_node, pos):
    """计算启发式估计值（欧氏距离）"""
    x1, y1 = pos[node]
    x2, y2 = pos[end_node]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def normalize(value, min_value, max_value, reverse=False):
    """归一化处理"""
    if max_value == min_value:
        return 0
    norm_value = (value - min_value) / (max_value - min_value)
    return 1 - norm_value if reverse else norm_value

class Car:
    def __init__(self, car_num, speed, start_position, end_position):
        """初始化车辆"""
        self.car_num = car_num
        self.start_position = start_position
        self.end_position = end_position
        self.current_position = start_position
        self.finished = False
        self.path = [start_position]  # 使用节点名称存储路径
        self.relative_time = 0
        self.speed = speed

    def move(self, G, current_time, heuristics, car_counts, attract_ranks, weights, pos):
        """模拟车辆移动"""
        if self.finished:
            return

        neighbors = list(G[self.current_position])
        if not neighbors:
            self.finished = True
            return

        next_node = None
        min_cost = float('inf')
        for neighbor in neighbors:
            if neighbor in self.path:
                continue  # 跳过已经走过的节点

            edge_length = G[self.current_position][neighbor]['length']
            cars_on_node = G.nodes[neighbor]['cars']
            attract_rank = G.nodes[neighbor]['attract_rank']

            edge_cost = normalize(edge_length, 0, max(nx.get_edge_attributes(G, 'length').values()))
            congestion_cost = normalize(cars_on_node, 0, max(car_counts))
            heuristic_cost = normalize(heuristic(neighbor, self.end_position, pos), 0, max(heuristics))
            attract_rank_cost = normalize(attract_rank, min(attract_ranks.values()), max(attract_ranks.values()), reverse=True)

            total_cost = (weights['edge'] * edge_cost +
                          weights['congestion'] * congestion_cost +
                          weights['heuristic'] * heuristic_cost +
                          weights['attract_rank'] * attract_rank_cost)
            if total_cost < min_cost:
                min_cost = total_cost
                next_node = neighbor

        if next_node is None:
            self.finished = True
            return

        self.current_position = next_node
        self.path.append(next_node)
        self.relative_time += G[self.path[-2]][self.path[-1]]['length'] / self.speed

        if self.current_position == self.end_position:
            self.finished = True

def simulate_specified_car(start_node, end_node, G, weights):
    """模拟指定车辆的路径"""
    pos = nx.get_node_attributes(G, 'pos')
    heuristics = {node: heuristic(node, end_node, pos) for node in G.nodes}
    attract_ranks = attract_rank(G)
    car_counts = [G.nodes[node]['cars'] for node in G.nodes]

    car = Car(1, 60, start_node, end_node)
    current_time = 0

    while not car.finished:
        car.move(G, current_time, heuristics, car_counts, attract_ranks, weights, pos)
        current_time += 1

    return car.path

def user_defined_path_selection(data_path, weights):
    """让用户选择起点和终点并模拟车辆路径"""
    # 从文件创建图
    G, dot = create_vertices(data_path)
    G = initialize_graph_from_dot(dot)
    compute_centrality(G)

    # 用户输入起点和终点
    start_node = int(input("请输入起点节点编号: "))
    end_node = int(input("请输入终点节点编号: "))

    # 模拟车辆并返回路径
    path = simulate_specified_car(start_node, end_node, G, weights)
    return path

# 示例数据，如何调用
# data_path = 'Vertices_Weight_create/node_data.xlsx'
# weights = {'edge': 0.5, 'congestion': 0.2, 'heuristic': 0.2, 'attract_rank': 0.1}
#
# # 执行用户输入选择，并打印结果
# path = user_defined_path_selection(data_path, weights)

