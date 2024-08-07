import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from Vertices_Weight_create.create_Vertices import edges,Vertices
# 节点及其坐标
Vertices =Vertices

# 边的连接关系
Edges = edges


# 初始化图
def initialize_graph(Vertices, Edges):
    G = nx.Graph()
    # 添加节点
    for node, pos in Vertices.items():
        G.add_node(node, pos=pos, cars=0)  # 初始化每个节点上的车辆数量为 0
    # 添加边并计算边的长度
    for edge in Edges:
        node1, node2 = edge
        pos1 = Vertices[node1]
        pos2 = Vertices[node2]
        length = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
        G.add_edge(node1, node2, length=length)

    return G


# 计算 PageRank 和中介中心性
def compute_centrality(G):
    pagerank_values = nx.pagerank(G, weight='length')
    betweenness_centrality_values = nx.betweenness_centrality(G, weight='length')

    # 将结果添加到图的节点属性中
    nx.set_node_attributes(G, pagerank_values, 'pagerank')
    nx.set_node_attributes(G, betweenness_centrality_values, 'betweenness')

    return pagerank_values, betweenness_centrality_values


# 定义 AttractRank 算法
def attract_rank(G, alpha=0.5, beta=0.5):
    """计算 AttractRank 值"""
    attractiveness = {}
    for node in G.nodes:
        pagerank_score = G.nodes[node]['pagerank']
        betweenness_score = G.nodes[node]['betweenness']
        attract_rank_score = alpha * pagerank_score + beta * betweenness_score
        attractiveness[node] = attract_rank_score

    # 将吸引力得分添加到图的节点属性中
    nx.set_node_attributes(G, attractiveness, 'attract_rank')

    return attractiveness


# 计算启发式估计值
def heuristic(node, end_node, pos):
    """计算启发式估计值（欧氏距离）"""
    x1, y1 = pos[node]
    x2, y2 = pos[end_node]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# 归一化处理
def normalize(value, min_value, max_value):
    """归一化函数"""
    if max_value == min_value:
        return 0  # 如果最小值等于最大值，避免除零错误
    return (value - min_value) / (max_value - min_value)


# 车辆类
class Car:
    def __init__(self, car_num, speed, start_position, end_position):
        self.car_num = car_num
        self.start_position = start_position
        self.end_position = end_position
        self.current_position = start_position
        self.finished = False
        self.path = [Vertices[start_position]]  # 使用坐标存储路径
        self.relative_time = 0
        self.speed = speed

    def move(self, G, current_time, heuristics, car_counts, attract_ranks, weights, pos):
        """模拟车辆移动"""
        if self.finished:
            return

        # 计算到达边终点的时间
        neighbors = list(G[self.current_position])
        if not neighbors:
            self.finished = True
            return

        next_node = None
        min_cost = float('inf')
        for neighbor in neighbors:
            if Vertices[neighbor] in self.path:
                continue  # 跳过已经走过的节点

            edge_length = G[self.current_position][neighbor]['length']
            cars_on_node = G.nodes[neighbor]['cars']
            attract_rank = G.nodes[neighbor]['attract_rank']

            # 归一化处理
            edge_cost = normalize(edge_length, 0, max(nx.get_edge_attributes(G, 'length').values()))
            congestion_cost = normalize(cars_on_node, 0, max(car_counts))
            heuristic_cost = normalize(heuristic(neighbor, self.end_position, pos), 0, max(heuristics))
            attract_rank_cost = normalize(attract_rank, min(attract_ranks.values()), max(attract_ranks.values()))

            # 计算总成本（加权和）
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

        # 更新车辆状态
        travel_time = (G[self.current_position][next_node]['length'] / self.speed) * (
                1 + G.nodes[next_node]['cars'] / 10.0)  # 考虑拥挤度影响
        self.relative_time += travel_time

        self.path.append(Vertices[next_node])  # 使用坐标存储路径
        self.current_position = next_node

        if self.current_position == self.end_position:
            self.finished = True
            return

        # 增加新节点上的车辆数量
        G.nodes[self.current_position]['cars'] += 1


# 初始化车辆
def initialize_cars(num_cars, G):
    cars = []
    for car_id in range(num_cars):
        start_node = random.choice(list(Vertices.keys()))
        end_node = random.choice(list(Vertices.keys()))
        while end_node == start_node:
            end_node = random.choice(list(Vertices.keys()))

        car = Car(car_id + 1, speed=8, start_position=start_node, end_position=end_node)
        cars.append(car)
        # 更新初始节点的车辆数量
        G.nodes[start_node]['cars'] += 1
    return cars


# 模拟车辆移动
def simulate_movement(cars, G, attract_rank_values, weights, pos):
    current_time = 0
    while not all(car.finished for car in cars):
        for car in cars:
            heuristics = [heuristic(node, car.end_position, pos) for node in pos]
            car_counts = [G.nodes[node]['cars'] for node in pos]
            car.move(G, current_time, heuristics, car_counts, attract_rank_values, weights, pos)
        current_time += 1


# 输出结果
def print_results(cars):
    for car in cars:
        print(f"车辆 {car.car_num}：")
        print(f"  起始点: {car.start_position} 坐标: {Vertices[car.start_position]}")
        print(f"  终点: {car.end_position} 坐标: {Vertices[car.end_position]}")
        print(f"  经过的路径: {car.path}")
        print(f"  总运行时间: {car.relative_time:.2f} 小时\n")


# 确定的权重
weights = {
    'edge': 0.19992673477200862,
    'congestion': 0.2032222127755475,
    'heuristic': 0.5307710705155291,
    'attract_rank': 0.06607998193691485
}


# 开始模拟
def start_simulation(num_cars, Vertices, Edges):
    G = initialize_graph(Vertices, Edges)
    pagerank_values, betweenness_centrality_values = compute_centrality(G)
    attract_rank_values = attract_rank(G)

    # 初始化车辆
    cars = initialize_cars(num_cars, G)

    # 模拟移动
    simulate_movement(cars, G, attract_rank_values, weights, Vertices)

    return cars

# 主程序
# num_cars = 10
# cars= start_simulation(num_cars, Vertices, Edges)
# print_results(cars)
