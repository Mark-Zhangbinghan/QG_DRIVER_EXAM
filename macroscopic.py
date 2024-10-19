import asyncio
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import numpy as np


# 自定义引用
from Vertices_Weight_create.create_Vertices import G
from end_dijkstra import run_simulation
from add_json import cars_to_json, cars_to_file, mat_hot_point, user_null_json, json_to_file
import share

# 数据引用
# import e自带data_path,weights
from road import data_path, e_weights
from road import user_defined_path_selection
macro_app = FastAPI()


# 推送车辆数量路由
@macro_app.put('/put_car')
async def put_car(get_params: Request):  # 要在url中写参数而不是请求体
    car_num = 50  # 预设被运算车辆的数量
    params = get_params.query_params
    car_num = params.get('car_num')
    if car_num and car_num.isdigit():  # 判断能否转换成整数
        car_num = int(car_num)  # 将字符串转换为整数
    else:
        car_num = -1  # 表示转换失败
    print(car_num)
    # 转换成功
    if car_num == -1:
        return {"need int"}
    else:
        # 根据接受到的car_num先计算宏观路径
        share.cars, share.weights = run_simulation(G=G, total_cars=car_num, round_num=5, speed=0.5)  # 直接计算path然后存成字典列表
        # 存成文件方便检查
        cars_to_file(share.cars)
        return {"put succeed"}


# 获取车辆路径路由
@macro_app.get("/get_path")
async def get_path():
    car_list = cars_to_json(share.cars, add_z=-3)
    print("car:")
    print("cnt/len")
    print(share.car_cnt + 1, "/", len(car_list))
    if share.car_cnt >= len(car_list):
        share.car_cnt = 0
    car_data = car_list[share.car_cnt]
    share.car_cnt += 1
    return car_data  # 直接返回字典


@macro_app.get("/get_weights")
async def get_weights():
    if share.weights_cnt < len(share.weights):
        weight_data = share.weights[share.weights_cnt]
    else:
        weights_cnt = 0
        weight_data = share.weights[weights_cnt]
    print("weights:")
    print("cnt/len")
    print(share.weights_cnt + 1, "/", len(share.weights))
    dot_json = mat_hot_point(weight_data)
    weight_pos = {
        "PosWeight": dot_json  # 修改成图形要的格式
    }
    share.weights_cnt += 1
    return weight_pos  # 直接返回字典


@macro_app.websocket("/ws_weights")
async def ws_weights(websocket: WebSocket):
    await websocket.accept()
    weight_cnt = 0
    try:
        while True:
            print(weight_cnt)
            print(len(share.weights))
            if weight_cnt >= len(share.weights):
                weight_cnt = 0  # 重置索引
            # 提取一次列表
            weight_data = share.weights[weight_cnt]
            # 转成json
            dot_json = mat_hot_point(weight_data)
            json_data = json.dumps(dot_json)
            await websocket.send_text(json_data)
            weight_cnt += 1
            await asyncio.sleep(0.5)
    except Exception as e:
        # 处理异常，例如连接关闭
        print(f"Websocket closed: {e}")


# 前端发请求和json运行用户自设路径
@macro_app.put("/put_user_path")
async def put_path(path_request: Request):
    path_json = await path_request.json()
    start_point = int(path_json["start_point"])
    end_point = int(path_json["end_point"])
    is_driving = path_json["is_driving"]  # 判断是否运行flag
    if is_driving == 1:
        user_path = user_defined_path_selection(data_path=data_path, weights=e_weights, start_node=start_point,
                                                end_node=end_point)  # 调用e函数求路径
        path_nodes = []
        for node in user_path:
            node_dict = {
                "x": node[0],
                "y": node[1],
                "z": 0
            }
            path_nodes.append(node_dict)
        path_pos = {
            "PathNodes": path_nodes  # 修改成图形要的格式
        }
    else:
        path_pos = user_null_json()  # 调用函数求空json
    filename = 'user_path.json'
    json_to_file(filename=filename, json_dict=path_pos)  # 调用函数写进文件
    if is_driving == 1:
        return {"message": "running successfully",
                "stage": 1}  # 成功则返回1
    else:
        return {"message": "stopped successfully",
                "stage": 0}  # 失败则返回0


# 图形重复发请求获取
@macro_app.get("/get_user_path")
async def get_user_path():
    filename = 'user_path.json'
    with open(filename, 'r', encoding='utf-8') as file:
        path_data = file.read()
        path_json = json.loads(path_data)
    return path_json
