import networkx as nx
import numpy as np
import random
import time
import queue
import threading
# from create_Vertices import create_vertices
# data_path = 'node_data.xlsx'
# G, dot = create_vertices(data_path)


def run_simulation(G, total_cars=10, round_num=5, speed=0.5):
    # 初始化图和其他相关参数
    road_data = [
        {'道路名称': G.edges[edge]['road'], '实际距离': G.edges[edge]['length']}
        for edge in G.edges
    ]

    pos = nx.get_node_attributes(G, 'pos')  # 获取每个节点的位置属性

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
        # 确保权重始终为正
        weight = d['length'] / (G.nodes[v]['attract_rank'] + 1e-6)
        return max(weight, 1e-6)

    class Car:
        def __init__(self, car_num, speed, start_position, end_position, path):
            self.car_num = car_num
            self.speed = speed
            self.start_position = start_position
            self.end_position = end_position
            self.path = path
            self.relative_time = 0.0

        def add_path_point(self, node_name, coords, travel_time):
            self.relative_time += travel_time
            self.path.append({
                'node_name': node_name,  # 添加节点名称
                'coords': coords,
                'relative_time': self.relative_time,
                'travel_time': travel_time,
                'timestamp': time.time()
            })

        def __lt__(self, other):
            return self.path[-1]['timestamp'] < other.path[-1]['timestamp']

    # class Car:
    #     def __init__(self, car_num, speed, start_position, end_position, path):
    #         self.car_num = car_num
    #         self.speed = speed
    #         self.start_position = start_position
    #         self.end_position = end_position
    #         self.path = path
    #         self.relative_time = 0.0
    #
    #     def add_path_point(self, coords, travel_time):
    #         self.relative_time += travel_time
    #         self.path.append({
    #             'coords': coords,
    #             'relative_time': self.relative_time,
    #             'travel_time': travel_time,
    #             'timestamp': time.time()
    #         })
    #
    #     def __lt__(self, other):
    #         return self.path[-1]['timestamp'] < other.path[-1]['timestamp']

    def calculate_stay_time(attractiveness):
        return max(0.1, attractiveness * 0.1)

    cars_info = []
    vertex_weight = []
    lock = threading.Lock()

    def simulate_vehicle_path(G, road_data, pos, car_num, output_queue):
        for _ in range(round_num):
            # 从road_data中随机选择起点和终点的道路
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
                continue  # 如果没有找到相连的边则跳过当前轮次

            start_edge = random.choice(start_connected_edges)
            end_edge = random.choice(end_connected_edges)

            try:
                # 使用Dijkstra算法计算路径，确保只使用图中存在的边
                path = nx.dijkstra_path(G, source=start_edge[0], target=end_edge[1],
                                        weight=lambda u, v, d: custom_weight(u, v, d, G))
            except nx.NetworkXNoPath:
                continue  # 如果没有找到路径则跳过当前轮
            car = Car(car_num, speed, pos[start_edge[0]], pos[end_edge[1]], [])
            car.add_path_point(start_edge[0], car.start_position, start_position / car.speed)  # 添加起始节点名称
            v = 1
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                edge_data = G[u][v]
                travel_time = edge_data['length'] / car.speed
                stay_time = calculate_stay_time(G.nodes[v]['attract_rank'])
                car.add_path_point(v, pos[v], travel_time + stay_time)  # 添加中间节点名称
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

            last_leg_time = end_position / car.speed
            car.add_path_point(path[-1], pos[path[-1]], last_leg_time)  # 添加终点节点名称

            # car = Car(car_num, speed, pos[start_edge[0]], pos[end_edge[1]], [])
            # car.add_path_point(car.start_position, start_position / car.speed)
            # v = 1
            # for i in range(len(path) - 1):
            #     u = path[i]
            #     v = path[i + 1]
            #     edge_data = G[u][v]
            #     travel_time = edge_data['length'] / car.speed
            #     stay_time = calculate_stay_time(G.nodes[v]['attract_rank'])
            #     car.add_path_point(pos[v], travel_time + stay_time)
            #     G.nodes[v]['weight'] += 1
            #     nx.set_node_attributes(G, attract_rank(G), 'attract_rank')
            #
            #     if i < len(path) - 2:
            #         try:
            #             new_path = nx.dijkstra_path(G, source=v, target=path[-1],
            #                                         weight=lambda u, v, d: custom_weight(u, v, d, G))
            #             if new_path != path[i + 1:]:
            #                 path = path[:i + 1] + new_path
            #         except nx.NetworkXNoPath:
            #             break
            #
            # last_leg_time = end_position / car.speed
            # car.add_path_point(pos[path[-1]], last_leg_time)

            with lock:
                vertex_weight.append({
                    node: {'weight': G.nodes[node]['weight'], 'pos': G.nodes[node]['pos']}
                    for node in G.nodes
                })

            G.nodes[v]['weight'] -= 1
            nx.set_node_attributes(G, attract_rank(G), 'attract_rank')

            output_queue.put((car.path[-1]['timestamp'], car))
            cars_info.append({
                'car_num': car.car_num,
                'speed': car.speed,
                'path': car.path
            })

            time.sleep(np.random.uniform(0.1, 0.3))

    output_queue = queue.PriorityQueue()

    threads = []
    for car_num in range(1, total_cars + 1):
        t = threading.Thread(target=simulate_vehicle_path, args=(G, road_data, pos, car_num, output_queue))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return cars_info, vertex_weight


# cars_info, vertex_weight = run_simulation(G, total_cars=10, round_num=5, speed=0.5)
# print(cars_info)
# print(vertex_weight)
# print(len(cars_info))
# print(len(vertex_weight))
