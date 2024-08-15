import json
import numpy as np
from CAV.code.starter import main


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

def mat_hot_point(weights):
    dot_list_json = []
    for weight_key, weight_value in weights.items():
        # 这里的权重列表中的每个条目是一个字典
        dot_dict = {
            "x": weight_value['pos'][0],
            "y": weight_value['pos'][1],
            "z": weight_value['weight']
        }
        dot_list_json.append(dot_dict)
    return dot_list_json


# 创建user_path的空json
def user_null_json():
    path_nodes = []
    node_dict = {
        "x": None,
        "y": None,
        "z": None
    }
    path_nodes.append(node_dict)  # 创建空json
    path_pos = {
        "PathNodes": path_nodes  # 修改成图形要的格式
    }
    return path_pos


# 存储json文件
def json_to_file(filename, json_dict):
    json_output = json.dumps(json_dict, indent=2)
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(json_output)
    print(f'path_json数据已成功写入到文件：{filename}')


def sub_path_json(pos):
    all_paths = []
    for car in range(pos.shape[1]):
        car_path = []
        for point in range(pos.shape[0]):
            x = pos[point, car, 0]
            y = 0.55
            z = pos[point, car, 1]
            car_path.append({"x": x, "y": y, "z": z})
        all_paths.append({"path": car_path})
    return all_paths


# 合并微观图列表然后保存到文件
def concatenate_arrays(arrays_list, file_name):
    # 过滤掉空列表，只保留numpy数组
    non_empty_arrays = [arr for arr in arrays_list if isinstance(arr, np.ndarray) and arr.size > 0]

    # 如果过滤后的数组列表为空，则跳过保存操作
    if not non_empty_arrays:
        print("没有非空的NumPy数组，跳过保存。")
        return

    # 检查所有非空数组的第一维和第三维形状是否一致
    first_shape = non_empty_arrays[0].shape[0]
    third_shape = non_empty_arrays[0].shape[2]
    for arr in non_empty_arrays:
        if arr.shape[0] != first_shape or arr.shape[2] != third_shape:
            raise ValueError("非空数组中存在形状不一致的情况")

    # 合并非空数组，沿着第二维（车辆数）进行合并
    concatenated_array = np.concatenate(non_empty_arrays, axis=1)

    # 四舍五入到三位小数
    rounded_array = np.round(concatenated_array, decimals=3)

    # 保存四舍五入后的数组
    np.save(file_name, rounded_array)
    print(f"已保存四舍五入到三位小数的合并后的数组到 {file_name}")
