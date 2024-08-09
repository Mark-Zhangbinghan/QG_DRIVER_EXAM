import networkx as nx
import numpy as np
import random
import time
import queue
import threading
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


def run_simulation(G, total_cars=10, round_num=5, speed=0.5):
    # 初始化图和其他相关参数


    road_data = [
        {'道路名称': edge[2], '实际距离': G.edges[edge[0], edge[1]]['length']}
        for edge in Edges
    ]

    pos = nx.get_node_attributes(G, 'pos')

    # 计算 PageRank 和中介中心性
    pagerank_values = nx.pagerank(G, weight='length')
    betweenness_centrality_values = nx.betweenness_centrality(G, weight='length')

    nx.set_node_attributes(G, pagerank_values, 'pagerank')
    nx.set_node_attributes(G, betweenness_centrality_values, 'betweenness')

    def attract_rank(G, alpha=0.5, beta=0.3, gamma=0.2):
        attractiveness = {}
        for node in G.nodes:
            pagerank_score = G.nodes[node]['pagerank']
            betweenness_score = G.nodes[node]['betweenness']
            node_weight = G.nodes[node]['weight']
            attract_rank_score = alpha * pagerank_score + beta * betweenness_score + gamma * node_weight
            attractiveness[node] = attract_rank_score
        return attractiveness

    attract_rank_values = attract_rank(G)
    nx.set_node_attributes(G, attract_rank_values, 'attract_rank')

    def custom_weight(u, v, d, G):
        return d['length'] / (G.nodes[v]['attract_rank'] + 1e-6)

    class Car:
        def __init__(self, car_num, speed, start_position, end_position, path):
            self.car_num = car_num
            self.speed = speed
            self.start_position = start_position
            self.end_position = end_position
            self.path = path
            self.relative_time = 0.0

        def add_path_point(self, coords, travel_time):
            self.relative_time += travel_time
            self.path.append({
                'coords': coords,
                'relative_time': self.relative_time,
                'travel_time': travel_time,
                'timestamp': time.time()
            })

        def __lt__(self, other):
            return self.path[-1]['timestamp'] < other.path[-1]['timestamp']

    def calculate_stay_time(attractiveness):
        return max(0.1, attractiveness * 0.1)

    cars_info = []

    def simulate_vehicle_path(G, road_data, pos, car_num, output_queue):
        for _ in range(round_num):
            start_road_index = np.random.choice(len(road_data))
            end_road_index = np.random.choice(len(road_data))

            while end_road_index == start_road_index:
                end_road_index = np.random.choice(len(road_data))

            start_road = road_data[start_road_index]
            end_road = road_data[end_road_index]
            start_position = np.random.uniform(0, start_road['实际距离'])
            end_position = np.random.uniform(0, end_road['实际距离'])

            start_connected_edges = [(u, v) for u, v, d in G.edges(data=True) if d['road'] == start_road['道路名称']]
            end_connected_edges = [(u, v) for u, v, d in G.edges(data=True) if d['road'] == end_road['道路名称']]

            if not start_connected_edges or not end_connected_edges:
                continue

            start_edge = random.choice(start_connected_edges)
            end_edge = random.choice(end_connected_edges)

            try:
                if start_position < end_position:
                    path = nx.dijkstra_path(G, source=start_edge[0], target=end_edge[1],
                                            weight=lambda u, v, d: custom_weight(u, v, d, G))
                else:
                    path = nx.dijkstra_path(G, source=start_edge[1], target=end_edge[0],
                                            weight=lambda u, v, d: custom_weight(u, v, d, G))
            except nx.NetworkXNoPath:
                continue

            car = Car(car_num, speed, pos[start_edge[0]], pos[end_edge[1]], [])
            car.add_path_point(car.start_position, start_position / car.speed)

            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                edge_data = G[u][v]
                travel_time = edge_data['length'] / car.speed
                stay_time = calculate_stay_time(G.nodes[v]['attract_rank'])
                car.add_path_point(pos[v], travel_time + stay_time)
                G.nodes[v]['weight'] += 1
                nx.set_node_attributes(G, attract_rank(G), 'attract_rank')

                if i < len(path) - 2:
                    try:
                        new_path = nx.dijkstra_path(G, source=v, target=path[-1],
                                                    weight=lambda u, v, d: custom_weight(u, v, d, G))
                        if new_path != path[i + 1:]:
                            path = path[:i + 1] + new_path
                    except nx.NetworkXNoPath:
                        break

                G.nodes[v]['weight'] -= 1
                nx.set_node_attributes(G, attract_rank(G), 'attract_rank')

            last_leg_time = end_position / car.speed
            car.add_path_point(pos[path[-1]], last_leg_time)

            output_queue.put((car.path[-1]['timestamp'], car))
            cars_info.append({
                'car_num': car.car_num,
                'speed': car.speed,
                'path': car.path
            })

            time.sleep(np.random.uniform(0.1, 0.3))

    # def print_results(output_queue):
    #     while True:
    #         try:
    #             timestamp, car = output_queue.get_nowait()
    #             print(f"车辆 {car.car_num} 到达终点:")
    #             print(f"  起始点位置: {car.start_position}")
    #             print(f"  结束点位置: {car.end_position}")
    #             for path_point in car.path:
    #                 timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(path_point['timestamp']))
    #                 print(
    #                     f"  到达坐标: {path_point['coords']}, 相对时间: {path_point['relative_time']:.2f}小时, 行驶时间: {path_point['travel_time']:.2f}小时, 时间戳: {timestamp_str}")
    #             print()
    #         except queue.Empty:
    #             break

    output_queue = queue.PriorityQueue()

    threads = []
    for car_num in range(1, total_cars + 1):
        t = threading.Thread(target=simulate_vehicle_path, args=(G, road_data, pos, car_num, output_queue))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # print_results(output_queue)

    return cars_info


# # 使用示例
# cars_info = run_simulation(G, total_cars=10, round_num=5, speed=0.5)
# print(f"总共保存了 {len(cars_info)} 辆车的信息:")
# print(cars_info)
