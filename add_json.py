import json
#
from 数模代码及数据集.数模代码及数据集.road import Car

Vertices = {'A': (4, 4), 'B': (18, 4), 'C': (32, 4), 'D': (4, 16), 'E': (16, 14), 'F': (28, 12), 'G': (4, 22),
            'H': (34, 38), 'I': (60, 34)}
Edges = [('A', 'B'), ('A', 'D'), ('B', 'C'), ('B', 'E'), ('C', 'F'), ('D', 'E'), ('D', 'G'), ('E', 'F'), ('E', 'H'),
         ('F', 'I'), ('G', 'H'), ('H', 'I')]

# class Car:
#     def __init__(self, car_num, speed, start_position, end_position):
#         self.car_num = car_num
#         self.speed = speed
#         self.start_position = start_position
#         self.end_position = end_position
#         self.path = []
#         self.relative_time = 0.0


# 假设这是你的车辆列表
cars = [
    Car(1, 8, 'A', 'F'),
    Car(2, 8, 'B', 'I')
]
cars[0].path = [(4, 4), (18, 4), (32, 4), ]
cars[1].path = []


def cars_to_json(cars_list):
    car_list_json = []
    for car in cars_list:
        car_dict = {
            "car_num": car.car_num,
            "speed": car.speed,
            "start_point": car.start_point,
            "end_point": car.end_point,
            "path": car.next_point
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
