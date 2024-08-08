import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import math
import queue
import threading
from dijkstrafun import simulate_vehicle_path, print_results, attract_rank

plt.rcParams['font.sans-serif'] = 'SimHei'

# 初始化顶点和边
Vertices = {'A': (4, 4), 'B': (18, 4), 'C': (32, 4), 'D': (4, 16), 'E': (16, 14), 'F': (28, 12), 'G': (4, 22), 'H': (34, 38), 'I': (60, 34)}
Edges = [('A', 'B', 'Road1'), ('A', 'D', 'Road2'), ('B', 'C', 'Road3'), ('B', 'E', 'Road4'), ('C', 'F', 'Road5'),
         ('D', 'E', 'Road6'), ('D', 'G', 'Road7'), ('E', 'F', 'Road8'), ('E', 'H', 'Road9'),
         ('F', 'I', 'Road10'), ('G', 'H', 'Road11'), ('H', 'I', 'Road12')]

# 定义节点权重
node_weights = {'A': 10, 'B': 12, 'C': 5, 'D': 8, 'E': 6, 'F': 4, 'G': 11, 'H': 9, 'I': 10}

# 创建图
G = nx.Graph()

# 添加顶点，并为每个顶点设置权重属性
for node, pos in Vertices.items():
    G.add_node(node, pos=pos, weight=node_weights[node])

# 添加边，并设置道路名称和实际距离（欧几里得距离）属性
for edge in Edges:
    G.add_edge(edge[0], edge[1], road=edge[2])
    # 计算两点之间的实际距离
    pos1 = Vertices[edge[0]]
    pos2 = Vertices[edge[1]]
    distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    G.edges[edge[0], edge[1]]['length'] = distance

# 生成 road_data 数据结构
road_data = pd.DataFrame([
    {'道路名称': edge[2], '实际距离': G.edges[edge[0], edge[1]]['length']}
    for edge in Edges
])

# 获取节点位置
pos = nx.get_node_attributes(G, 'pos')

# 绘制图
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
plt.show()

# 计算 PageRank 和中介中心性
pagerank_values = nx.pagerank(G, weight='length')
betweenness_centrality_values = nx.betweenness_centrality(G, weight='length')

# 将结果添加到图的节点属性中
nx.set_node_attributes(G, pagerank_values, 'pagerank')
nx.set_node_attributes(G, betweenness_centrality_values, 'betweenness')

# 计算 AttractRank 值
attract_rank_values = attract_rank(G)

# 将 AttractRank 结果添加到图的节点属性中
nx.set_node_attributes(G, attract_rank_values, 'attract_rank')

# 创建优先队列用于存储输出
output_queue = queue.PriorityQueue()

# 保存车辆信息的列表
cars_info = []

# 启动打印线程
print_thread = threading.Thread(target=print_results, args=(output_queue,))
print_thread.start()

# 为每辆车创建一个线程并开始模拟
threads = []
for car_num in range(1, 11):
    t = threading.Thread(target=simulate_vehicle_path,
                         args=(G, road_data, pos, car_num, output_queue, cars_info))
    threads.append(t)
    t.start()

# 等待所有车辆线程完成
for t in threads:
    t.join()

# 发送结束信号
output_queue.put((float('inf'), None, None))
# 等待打印线程完成
print_thread.join()

# 打印保存的车辆信息
print(f"总共保存了 {len(cars_info)} 辆车的信息:")
print(cars_info)
