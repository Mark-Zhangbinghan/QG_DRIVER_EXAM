class Vertex():
    def __init__(self, name, dot_weight, dot_place, near_dot):
        self.name = name
        self.dot_weight = dot_weight
        self.dot_place = dot_place
        self.near_dot = near_dot

Vertices = {'A': (4, 4), 'B': (18, 4), 'C': (32, 4), 'D': (4, 16), 'E': (16, 14), 'F': (28, 12), 'G': (4, 22), 'H': (34, 38), 'I': (60, 34)}
edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('B', 'E'), ('C', 'F'), ('D', 'E'), ('D', 'G'), ('E', 'F'), ('E', 'H'), ('F', 'I'), ('G', 'H'), ('H', 'I')]
dot = []
for vertex in Vertices.items():
    near = []
    for edge in edges:
        if vertex[0] == edge[0]:
            near.append(edge[1])
    dot.append(Vertex(vertex[0], 1, vertex[1], near))
for i in range(len(dot)):
    print(dot[i].name, dot[i].dot_weight, dot[i].dot_place, dot[i].near_dot)

