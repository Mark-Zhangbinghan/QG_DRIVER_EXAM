import asyncio
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi import Request
import uvicorn
import json
import numpy as np
# 自定函数
from end_dijkstra import run_simulation
from Vertices_Weight_create.create_Vertices import G
from add_json import cars_to_json, cars_to_file, mat_hot_point, user_null_json, json_to_file, sub_path_json, \
    concatenate_arrays
# import e自带data_path,weights
from road import data_path, e_weights
from road import user_defined_path_selection
from CAV.code.starter import main

app = FastAPI()
car_cnt = 0  # 车辆计数器
weights_cnt = 0
cars = []  # 全局列表cars
weights = []  # 全局列表weight
# 初始化user_path.json
origin_path = user_null_json()  # 创建空json
origin_filename = 'user_path.json'
json_to_file(filename=origin_filename, json_dict=origin_path)  # 调用函数写进文件
print("初始化user_path.json成功")

# 使用main算出路径参数

'''all_arrays = main()
no_n_arrays = all_arrays[:len(all_arrays) // 2]
with_n_arrays = all_arrays[len(all_arrays) // 2:]

concatenate_arrays(no_n_arrays, 'concatenated_no_n.npy')
concatenate_arrays(with_n_arrays, 'concatenated_with_n.npy')'''
# 保存带 n 的数组
concatenated_no_n = np.load('concatenated_no_n.npy', allow_pickle=True)

# 读取带 n 的合并后的数组
concatenated_with_n = np.load('concatenated_with_n.npy', allow_pickle=True)

# 打印数组的shape来验证
print("不带 n 的数组形状:", concatenated_no_n.shape)
print("带 n 的数组形状:", concatenated_with_n.shape)
no_n_num = concatenated_no_n.shape[1]
n_num = concatenated_with_n.shape[1]
# pos_v11, pos_v12, pos_v11n, pos_v12n = main()
# # 第一轮
# all_path_v11 = sub_path_json(pos_v11)
# v11_num = len(all_path_v11)
# all_path_v12 = sub_path_json(pos_v12)
# v12_num = len(all_path_v12)
# # 第二轮
# all_path_v11n = sub_path_json(pos_v11n)
# v11n_num = len(all_path_v11n)
# all_path_v12n = sub_path_json(pos_v12n)
# v12n_num = len(all_path_v12n)
sub_car_cnt = 0  # 微观图车辆计数器


# 判断连接是否成功路由
@app.get('/')
@app.post('/')
@app.put('/')
@app.delete('/')
async def read_root():
    return {"Connect Succeed"}


# 推送车辆数量路由
@app.put('/put_car')
async def put_car(get_params: Request):  # 要在url中写参数而不是请求体
    global cars
    global weights
    car_num = 10  # 预设被运算车辆的数量
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
        cars, weights = run_simulation(G=G, total_cars=car_num, round_num=1, speed=0.5)  # 直接计算path然后存成字典列表
        # 存成文件方便检查
        cars_to_file(cars)
        for weight in weights:
            print(weight)
        return {"put succeed"}


# 获得微观图路口数量
@app.put('/put_path_num')
async def put_path_num(get_params: Request):  # 要在url中写参数而不是请求体
    path_num = 4  # 预设岔路的数量
    params = get_params.query_params
    path_num = params.get('path_num')
    if path_num and path_num.isdigit():  # 判断能否转换成整数
        path_num = int(path_num)  # 将字符串转换为整数
    else:
        path_num = -1  # 表示转换失败
    print(path_num)
    if path_num == -1:
        return {"need int"}
    else:
        return {"put succeed"}


# 获取车辆路径路由
@app.get("/get_path")
async def get_path():  # 要在body中写参数
    global car_cnt
    car_list = cars_to_json(cars, add_z=-3)
    print("car:")
    print("cnt/len")
    print(car_cnt + 1, "/", len(car_list))
    if car_cnt >= len(car_list):
        car_cnt = 0
    car_data = car_list[car_cnt]
    car_cnt += 1
    return car_data  # 直接返回字典


@app.get("/get_weights")
async def get_weights():
    global weights_cnt
    if weights_cnt >= len(weights):
        weights_cnt = 0
    weight_data = weights[weights_cnt]
    print("weights:")
    print("cnt/len")
    print(weights_cnt + 1, "/", len(weights))
    dot_json = mat_hot_point(weight_data)
    weight_pos = {
        "PosWeight": dot_json  # 修改成图形要的格式
    }
    weights_cnt += 1
    return weight_pos  # 直接返回字典


@app.websocket("/ws_weights")
async def ws_weights(websocket: WebSocket):
    await websocket.accept()
    weight_cnt = 0
    try:
        while True:
            print(weight_cnt)
            print(len(weights))
            if weight_cnt >= len(weights):
                weight_cnt = 0  # 重置索引
            # 提取一次列表
            weight_data = weights[weight_cnt]
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
@app.put("/put_user_path")
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
        return {"running successfully"}
    else:
        return {"stopped successfully"}


# 图形重复发请求获取
@app.get("/get_user_path")
async def get_user_path():
    filename = 'user_path.json'
    with open(filename, 'r', encoding='utf-8') as file:
        path_data = file.read()
        path_json = json.loads(path_data)
    return path_json


@app.get("/get_sub_num")
async def get_sub_num():
    num_dict = {
        "x": no_n_num,
        "y": n_num
    }
    return num_dict


@app.get("/get_sub_path")
async def get_sub_path():
    no_n_list = sub_path_json(concatenated_no_n)
    n_list = sub_path_json(concatenated_with_n)
    global sub_car_cnt
    switch_mode = 0  # 0是红灯前,1是红灯后
    if sub_car_cnt >= no_n_num and switch_mode == 0:
        switch_mode = 1  # 切换模式
        sub_car_cnt = 0
    if sub_car_cnt >= n_num and switch_mode == 1:
        switch_mode = 0
        sub_car_cnt = 0
    print("sub_car:")
    print("cnt/len")
    if switch_mode == 0:
        print(sub_car_cnt + 1, "/", no_n_num)
        sub_car_json = no_n_list[sub_car_cnt]
    else:
        print(sub_car_cnt + 1, "/", n_num)
        sub_car_json = n_list[sub_car_cnt]
    return sub_car_json


# 主监听函数
if __name__ == "__main__":
    # uvicorn.run(app="web:app", host="192.168.0.92", port=8080, reload=False)#华为云
    uvicorn.run(app="web:app", host="127.0.0.1", port=8080, reload=False)
