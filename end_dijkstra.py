import networkx as nx
import numpy as np
import random
import time
import queue
import threading
# data_path = 'node_data.xlsx'
# G, dot = get_graph_and_vertices(data_path)
def run_simulation(G, total_cars=10, round_num=5, speed=0.5):
    # 初始化图和其他相关参数
    road_data = [
        {'道路名称': G.edges[edge]['road'], '实际距离': G.edges[edge]['length']}
        for edge in G.edges
    ]

    pos = nx.get_node_attributes(G, 'pos')  # 获取每个节点的位置属性

    # 计算 PageRank 和中介中心性
    pagerank_values = nx.pagerank(G, weight='length')  # 计算 PageRank 值
    betweenness_centrality_values = nx.betweenness_centrality(G, weight='length')  # 计算中介中心性

    nx.set_node_attributes(G, pagerank_values, 'pagerank')  # 设置节点的 PageRank 属性
    nx.set_node_attributes(G, betweenness_centrality_values, 'betweenness')  # 设置节点的中介中心性属性

    def attract_rank(G, alpha=0.5, beta=0.3, gamma=0.2):
        # 计算每个节点的吸引力排名
        attractiveness = {}
        for node in G.nodes:
            pagerank_score = G.nodes[node]['pagerank']  # 获取节点的 PageRank 分数
            betweenness_score = G.nodes[node]['betweenness']  # 获取节点的中介中心性分数
            node_weight = G.nodes[node]['weight']  # 获取节点的权重
            attract_rank_score = alpha * pagerank_score + beta * betweenness_score + gamma * node_weight  # 计算吸引力分数
            attractiveness[node] = attract_rank_score  # 存储吸引力分数
        return attractiveness

    attract_rank_values = attract_rank(G)  # 计算每个节点的吸引力排名
    nx.set_node_attributes(G, attract_rank_values, 'attract_rank')  # 设置节点的吸引力排名属性

    def custom_weight(u, v, d, G):
        # 自定义的权重函数，用于路径选择
        return d['length'] / (G.nodes[v]['attract_rank'] + 1e-6)

    class Car:
        def __init__(self, car_num, speed, start_position, end_position, path):
            self.car_num = car_num  # 车辆编号
            self.speed = speed  # 车辆速度
            self.start_position = start_position  # 起始位置
            self.end_position = end_position  # 终点位置
            self.path = path  # 路径
            self.relative_time = 0.0  # 相对时间

        def add_path_point(self, coords, travel_time):
            # 向路径中添加一个新点
            self.relative_time += travel_time  # 更新相对时间
            self.path.append({
                'coords': coords,  # 当前坐标
                'relative_time': self.relative_time,  # 当前的相对时间
                'travel_time': travel_time,  # 当前段的行驶时间
                'timestamp': time.time()  # 当前时间戳
            })

        def __lt__(self, other):
            # 比较两个车辆路径中的最后一个时间戳
            return self.path[-1]['timestamp'] < other.path[-1]['timestamp']

    def calculate_stay_time(attractiveness):
        # 计算车辆在节点的停留时间
        return max(0.1, attractiveness * 0.1)

    cars_info = []  # 存储车辆信息
    vertex_weight = []  # 存储节点权重变化信息
    lock = threading.Lock()  # 线程锁

    def simulate_vehicle_path(G, road_data, pos, car_num, output_queue):
        # 模拟单个车辆的路径
        for _ in range(round_num):
            start_road_index = np.random.choice(len(road_data))  # 随机选择起始道路
            end_road_index = np.random.choice(len(road_data))  # 随机选择终点道路

            while end_road_index == start_road_index:
                end_road_index = np.random.choice(len(road_data))  # 确保起始和终点不同

            start_road = road_data[start_road_index]  # 获取起始道路信息
            end_road = road_data[end_road_index]  # 获取终点道路信息
            start_position = np.random.uniform(0, start_road['实际距离'])  # 随机生成起始位置
            end_position = np.random.uniform(0, end_road['实际距离'])  # 随机生成终点位置

            start_connected_edges = [(u, v) for u, v, d in G.edges(data=True) if d['road'] == start_road['道路名称']]  # 包含所有与起始道路名称匹配的边的列表
            end_connected_edges = [(u, v) for u, v, d in G.edges(data=True) if d['road'] == end_road['道路名称']]  # 包含所有与终点道路名称匹配的边的列表

            if not start_connected_edges or not end_connected_edges:
                continue  # 如果没有找到相连的边则跳过当前轮次

            start_edge = random.choice(start_connected_edges)  # 随机选择起始边
            end_edge = random.choice(end_connected_edges)  # 随机选择终点边

            try:
                if start_position < end_position:
                    path = nx.dijkstra_path(G, source=start_edge[0], target=end_edge[1],
                                            weight=lambda u, v, d: custom_weight(u, v, d, G))  # 使用 Dijkstra 算法计算路径
                else:
                    path = nx.dijkstra_path(G, source=start_edge[1], target=end_edge[0],
                                            weight=lambda u, v, d: custom_weight(u, v, d, G))  # 使用 Dijkstra 算法计算路径
            except nx.NetworkXNoPath:
                continue  # 如果没有找到路径则跳过当前轮次

            car = Car(car_num, speed, pos[start_edge[0]], pos[end_edge[1]], [])  # 创建 Car 对象
            car.add_path_point(car.start_position, start_position / car.speed)  # 添加路径起点
            v = 0
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                edge_data = G[u][v]  # 获取边的数据
                travel_time = edge_data['length'] / car.speed  # 计算行驶时间
                stay_time = calculate_stay_time(G.nodes[v]['attract_rank'])  # 计算停留时间
                car.add_path_point(pos[v], travel_time + stay_time)  # 添加路径点
                G.nodes[v]['weight'] += 1  # 增加节点权重
                nx.set_node_attributes(G, attract_rank(G), 'attract_rank')  # 更新吸引力排名

                if i < len(path) - 2:
                    try:
                        new_path = nx.dijkstra_path(G, source=v, target=path[-1],
                                                    weight=lambda u, v, d: custom_weight(u, v, d, G))  # 重新计算路径
                        if new_path != path[i + 1:]:
                            path = path[:i + 1] + new_path  # 更新路径
                    except nx.NetworkXNoPath:
                        break  # 如果没有找到路径则跳出循环

            last_leg_time = end_position / car.speed  # 计算最后一段的行驶时间
            car.add_path_point(pos[path[-1]], last_leg_time)  # 添加路径终点

            with lock:
                vertex_weight.append({
                    node: {'weight': G.nodes[node]['weight'], 'pos': G.nodes[node]['pos']}
                    for node in G.nodes
                })  # 记录节点权重变化

            G.nodes[v]['weight'] -= 1  # 减少节点权重
            nx.set_node_attributes(G, attract_rank(G), 'attract_rank')  # 更新吸引力排名

            output_queue.put((car.path[-1]['timestamp'], car))  # 将车辆路径添加到队列中
            cars_info.append({
                'car_num': car.car_num,
                'speed': car.speed,
                'path': car.path
            })  # 将车辆信息存储在 cars_info 中

            time.sleep(np.random.uniform(0.1, 0.3))  # 随机等待时间模拟现实情况

    output_queue = queue.PriorityQueue()  # 创建优先级队列

    threads = []
    for car_num in range(1, total_cars + 1):
        t = threading.Thread(target=simulate_vehicle_path, args=(G, road_data, pos, car_num, output_queue))
        threads.append(t)
        t.start()  # 启动每个线程，模拟每辆车的行驶路径

    for t in threads:
        t.join()  # 等待所有线程完成

    return cars_info, vertex_weight  # 返回车辆信息和节点权重变化信息



# cars_info, vertex_weight = run_simulation(G, total_cars=10, round_num=5, speed=0.5)
# print(cars_info)
# print(vertex_weight
# print(len(cars_info))
# print(len(vertex_weight))