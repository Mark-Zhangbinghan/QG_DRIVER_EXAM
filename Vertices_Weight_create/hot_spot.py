import networkx as nx
import math
import numpy as np
import draw_map


# 计算每个结点对于全局的影响度
def hot_spot(G):
    alpha = 0.1

    # 计算 Katz Centrality
    try:
        katz_centrality = nx.katz_centrality(G, alpha=alpha, max_iter=10000)
    except nx.exception.PowerIterationFailedConvergence as e:
        print("Power iteration failed to converge:", e)
    # 计算pagerank
    pagerank = nx.pagerank(G)

    # 两种算法开始结合
    combined_centrality = {node: 0.5 * katz_centrality[node] + 0.5 * pagerank[node] for node in G.nodes()}
    # 输出结合后的效果
    for node, cent in sorted(combined_centrality.items(), key=lambda item: item[1], reverse=True):
        print(f"{node} {cent:.4f}")



"""
# 建立马尔可夫矩阵
def create_markof(G):
    n = len(G.nodes())  # 图结点的个数(矩阵的边长)
    # 建立邻接稀疏矩阵
    adjacency = nx.adjacency_matrix(G)
    # 计算出度值
    degrees = dict(G.degree())
    # 建立马尔可夫矩阵
    markof = np.zeros((n, n))
    for edge in create_Vertices.Edges:
        markof[list(create_Vertices.Vertices.keys()).index(edge[0]), 
        list(create_Vertices.Vertices.keys()).index(edge[1])] = 1
        markof[list(create_Vertices.Vertices.keys()).index(edge[1]), 
        list(create_Vertices.Vertices.keys()).index(edge[0])] = 1

    for i in range(n):
        if degrees[list(create_Vertices.Vertices.keys())[i]] != 0:
            markof[:, i] /= degrees[list(create_Vertices.Vertices.keys())[i]]
    return markof


# 建立邻接矩阵
def make_adjacency(Vertices, Edges):
    # 计算顶点的数量
    n = len(Vertices)
    # 构造全零的邻接矩阵
    adjacency = np.zeros((n, n), dtype=int)

    # 将顶点映射到邻接矩阵的索引
    vertex_to_index = {vertex: index for index, vertex in enumerate(Vertices)}

    # 填充邻接矩阵
    for edge in Edges:
        u, v = vertex_to_index[edge[0]], vertex_to_index[edge[1]]
        adjacency[u][v] = 1
        adjacency[v][u] = 1
    return adjacency


adjacency = make_adjacency(Vertices, Edges)
print(adjacency)
"""
