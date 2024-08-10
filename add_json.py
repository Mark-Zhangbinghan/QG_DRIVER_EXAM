import json


# class Car:
#     def __init__(self, car_num, speed, start_position, end_position):
#         self.car_num = car_num
#         self.speed = speed
#         self.start_position = start_position
#         self.end_position = end_position
#         self.path = []
#         self.relative_time = 0.0

# e版本
# cars = start_simulation(10, vertices, edges)

def cars_to_file(cars_list, add_z=-3):
    car_list_json = []
    for car in cars_list:
        # 转换路径为所需的格式
        path_list = [{"x": point['coords'][0], "y": point['coords'][1], "z": add_z} for point in car['path']]

        # 构建车辆的字典
        car_dict = {
            "car_num": car['car_num'],
            "speed": car['speed'],
            "path": path_list
        }
        car_list_json.append(car_dict)
    # 转成json
    json_output = json.dumps({"CarList": car_list_json}, indent=2)
    filename = 'cars_data.json'
    # 打开文件，准备写入
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(json_output)
    print(f'JSON数据已成功写入到文件：{filename}')


def cars_to_json(cars_list, add_z=-3):
    car_list_json = []
    for car in cars_list:
        # 转换路径为所需的格式
        path_list = [{"x": point['coords'][0], "y": point['coords'][1], "z": add_z} for point in car['path']]

        # 构建车辆的字典
        car_dict = {
            "speed": car['speed'],
            "path": path_list
        }
        car_list_json.append(car_dict)
    return car_list_json


# cars = run_simulation(G=G, total_cars=10, round_num=1, speed=0.5)
# cars_to_json(cars)

def mat_hot_point(dot):
    dot_list_json = []
    for the_dot in dot:
        # dot字典
        dot_dict = {
            "x": the_dot.dot_place[0],
            "y": the_dot.dot_place[1],
            "weight": the_dot.dot_weight
        }
        dot_list_json.append(dot_dict)
    return dot_list_json
