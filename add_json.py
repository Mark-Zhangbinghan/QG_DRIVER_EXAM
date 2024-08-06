import json


class Car:
    def __init__(self, car_num, speed, start_point, end_point, next_point):
        self.car_num = car_num
        self.speed = speed
        self.start_point = start_point
        self.end_point = end_point
        self.next_point = next_point


# 假设这是你的车辆列表
cars = [
    Car(1, 1, {'x': 1, 'y': 1}, {'x': 1, 'y': 1}, {'x': 1, 'y': 1}),
    Car(2, 1, {'x': 1, 'y': 1}, {'x': 1, 'y': 1}, {'x': 1, 'y': 1})
]


def cars_to_json(cars_list):
    car_list_json = []

    for car in cars_list:
        car_dict = {
            "car_num": car.car_num,
            "speed": car.speed,
            "start_point": car.start_point,
            "end_point": car.end_point,
            "next_point": car.next_point
        }
        car_list_json.append(car_dict)

    # 将车辆列表字典转换为JSON字符串
    return json.dumps({"CarList": car_list_json}, indent=2)


# 使用函数将车辆列表转换为JSON
json_output = cars_to_json(cars)
print(json_output)


filename = 'cars_data.json'

# 打开文件，准备写入
with open(filename, 'w', encoding='utf-8') as file:
    file.write(json_output)

print(f'JSON数据已成功写入到文件：{filename}')