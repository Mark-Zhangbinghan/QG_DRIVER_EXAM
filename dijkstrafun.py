import networkx as nx
import numpy as np
import random
import time

# 定义车辆类
class Car:
    def __init__(self, car_num, speed, start_position, end_position, path):
        self.car_num = car_num  # 车辆编号
        self.speed = speed  # 车辆速度
        self.start_position = start_position  # 起始位置
        self.end_position = end_position  # 结束位置
        self.path = path  # 行驶路径
        self.relative_time = 0.0  # 相对时间

    # 添加路径点
    def add_path_point(self, coords, travel_time):
        self.relative_time += travel_time  # 更新相对时间
        self.path.append({
            'coords': coords,  # 当前坐标
            'relative_time': self.relative_time,  # 相对时间
            'travel_time': travel_time,  # 行驶时间
            'timestamp': time.time()  # 时间戳
        })

    # 定义小于（<）运算符，用于车辆之间的比较
    def __lt__(self, other):
        return self.path[-1]['timestamp'] < other.path[-1]['timestamp']

# 基于吸引力计算停留时间的函数
def calculate_stay_time(attractiveness):
    return max(0.1, attractiveness * 0.1)  # 最小停留时间为0.1小时

# 定义 AttractRank 算法
def attract_rank(G, alpha=0.5, beta=0.3, gamma=0.2):
    attractiveness = {}
    for node in G.nodes:
        pagerank_score = G.nodes[node]['pagerank']  # PageRank 分数
        betweenness_score = G.nodes[node]['betweenness']  # 中介中心性分数
        node_weight = G.nodes[node]['weight']  # 节点权重
        attract_rank_score = alpha * pagerank_score + beta * betweenness_score + gamma * node_weight
        attractiveness[node] = attract_rank_score  # 计算 AttractRank 分数
    return attractiveness

# 根据道路长度和吸引力分数自定义权重计算
def custom_weight(u, v, d, G):
    return d['length'] / (G.nodes[v]['attract_rank'] + 1e-6)

# 模拟车辆行驶路径
def simulate_vehicle_path(G, road_data, pos, car_num, output_queue, cars_info):
    uniform_speed = np.random.uniform(0.3, 0.7)  # 车辆速度
    round_num = 0  # 回合计数
    while round_num < 5:  # 循环五次
        round_num += 1
        start_road_index = np.random.choice(road_data.index)
        end_road_index = np.random.choice(road_data.index)

        while end_road_index == start_road_index:
            end_road_index = np.random.choice(road_data.index)

        start_road = road_data.iloc[start_road_index]
        end_road = road_data.iloc[end_road_index]
        start_road_name = start_road['道路名称']
        end_road_name = end_road['道路名称']
        start_position = np.random.uniform(0, start_road['实际距离'])
        end_position = np.random.uniform(0, end_road['实际距离'])

        start_connected_edges = [(u, v) for u, v, d in G.edges(data=True) if d['road'] == start_road_name]
        end_connected_edges = [(u, v) for u, v, d in G.edges(data=True) if d['road'] == end_road_name]

        if not start_connected_edges or not end_connected_edges:
            continue

        start_edge = random.choice(start_connected_edges)
        end_edge = random.choice(end_connected_edges)

        try:
            if start_position < end_position:
                path = nx.dijkstra_path(G, source=start_edge[0], target=end_edge[1], weight=lambda u, v, d: custom_weight(u, v, d, G))
            else:
                path = nx.dijkstra_path(G, source=start_edge[1], target=end_edge[0], weight=lambda u, v, d: custom_weight(u, v, d, G))
        except nx.NetworkXNoPath:
            continue

        car = Car(car_num, uniform_speed, pos[start_edge[0]], pos[end_edge[1]], [])  # 创建 Car 对象

        first_leg_time = start_position / car.speed
        car.add_path_point(car.start_position, first_leg_time)  # 添加路径点

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            edge_data = G[u][v]
            travel_time = edge_data['length'] / car.speed

            stay_time = calculate_stay_time(G.nodes[v]['attract_rank'])
            car.add_path_point(pos[v], travel_time + stay_time)  # 添加路径点并更新停留时间

            G.nodes[v]['weight'] += 1
            nx.set_node_attributes(G, attract_rank(G), 'attract_rank')  # 更新 AttractRank

            if i < len(path) - 2:
                try:
                    new_path = nx.dijkstra_path(G, source=v, target=path[-1], weight=lambda u, v, d: custom_weight(u, v, d, G))
                    if new_path != path[i + 1:]:
                        path = path[:i + 1] + new_path
                except nx.NetworkXNoPath:
                    break

            G.nodes[v]['weight'] -= 1
            nx.set_node_attributes(G, attract_rank(G), 'attract_rank')  # 重新设置 AttractRank

        last_leg_time = end_position / car.speed
        car.add_path_point(pos[path[-1]], last_leg_time)  # 添加最后一段路径点

        output_queue.put((car.path[-1]['timestamp'], car, round_num))  # 输出队列
        cars_info.append({
            'car_num': car.car_num,
            'speed': car.speed,
            'path': car.path
        })

        time.sleep(np.random.uniform(1, 3))

# 打印输出队列中的结果
def print_results(output_queue):
    while True:
        timestamp, car, round_num = output_queue.get()
        if car is None:  # 结束信号
            break
        print(f"车辆 {car.car_num} 在第 {round_num} 轮到达终点:")
        print(f"  起始点位置: {car.start_position}")
        print(f"  结束点位置: {car.end_position}")
        for path_point in car.path:
            timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(path_point['timestamp']))
            print(
                f"  到达坐标: {path_point['coords']}, 相对时间: {path_point['relative_time']:.2f}小时, 行驶时间: {path_point['travel_time']:.2f}小时, 时间戳: {timestamp_str}")
        print()
