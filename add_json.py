import json
#
from Vertices_Weight_create.create_Vertices import edges, vertices
from jicheng_fun import run_simulation
from jicheng_fun import G

# 节点及其坐标
vertices = vertices

# 边连接关系
edges = edges

# class Car:
#     def __init__(self, car_num, speed, start_position, end_position):
#         self.car_num = car_num
#         self.speed = speed
#         self.start_position = start_position
#         self.end_position = end_position
#         self.path = []
#         self.relative_time = 0.0

# cars = start_simulation(10, vertices, edges)
cars = run_simulation(G=G, total_cars=10, round_num=1, speed=0.5)
for car in cars:
    print(car['car_num'])


def cars_to_json(cars_list):
    car_list_json = []
    for car in cars_list:
        # 转换路径为所需的格式
        path_list = [{"x": point['coords'][0], "y": point['coords'][1]} for point in car['path']]

        # 构建车辆的字典
        car_dict = {
            "car_num": car['car_num'],
            "speed": car['speed'],
            "path": path_list
        }
        car_list_json.append(car_dict)
    # 转成json
    json_output = json.dumps({"CarList": car_list_json}, indent=2)

    print(json_output)
    filename = 'cars_data.json'
    # 打开文件，准备写入
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(json_output)
    print(f'JSON数据已成功写入到文件：{filename}')
    return json_output

# cars_to_json(cars)
