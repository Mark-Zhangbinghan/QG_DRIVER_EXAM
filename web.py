import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from macroscopic import macro_app
from microscopic import micro_app
import uvicorn
import numpy as np
from multiprocessing import Process

# 自定义函数
from Vertices_Weight_create.create_Vertices import G
from add_json import mat_hot_point, user_null_json, json_to_file, sub_path_json, concatenate_arrays, sub_switch_road
from CAV.code.starter import main
from CAV.code.starter_Tjunc import t_main

# 创建宏观图应用
app_macro = FastAPI()
# 创建微观图应用
app_micro = FastAPI()


# 初始化 CORS 中间件配置
def setup_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有来源
        allow_credentials=True,  # 允许携带凭证信息
        allow_methods=["*"],  # 允许所有方法
        allow_headers=["*"],  # 允许所有头部
    )


# 为两个应用分别设置 CORS
setup_cors(app_macro)
setup_cors(app_micro)

# 挂载子应用
app_macro.mount("/macroscopic", macro_app)
app_micro.mount("/microscopic", micro_app)

# 初始化 user_path.json
origin_path = user_null_json()  # 创建空json
origin_filename = 'user_path.json'
json_to_file(filename=origin_filename, json_dict=origin_path)  # 调用函数写进文件
print("初始化 user_path.json 成功")

# 十字路口文件
# all_arrays = t_main()
# no_n_arrays = all_arrays[:len(all_arrays) // 2]
# with_n_arrays = all_arrays[len(all_arrays) // 2:]
# # 保存带 n 的数组
# concatenate_arrays(no_n_arrays, 't_concatenated_no_n.npy')
# concatenate_arrays(with_n_arrays, 't_concatenated_with_n.npy')
# 三岔路口文件
# all_arrays = t_main()
# no_n_arrays = all_arrays[:len(all_arrays) // 2]
# with_n_arrays = all_arrays[len(all_arrays) // 2:]
# # 保存带 n 的数组
# concatenate_arrays(no_n_arrays, 't_concatenated_no_n.npy')
# concatenate_arrays(with_n_arrays, 't_concatenated_with_n.npy')


# 读取带 n 的合并后的数组
# 十字路口
concatenated_no_n = np.load('concatenated_no_n.npy', allow_pickle=True)
concatenated_with_n = np.load('concatenated_with_n.npy', allow_pickle=True)
# 打印数组的shape来验证
print("不带 n 的数组形状:", concatenated_no_n.shape)
print("带 n 的数组形状:", concatenated_with_n.shape)
no_n_num = concatenated_no_n.shape[1]
n_num = concatenated_with_n.shape[1]

# 三岔路口
t_concatenated_no_n = np.load('t_concatenated_no_n.npy', allow_pickle=True)
t_concatenated_with_n = np.load('t_concatenated_with_n.npy', allow_pickle=True)
print("t不带 n 的数组形状:", concatenated_no_n.shape)
print("t带 n 的数组形状:", concatenated_with_n.shape)
t_no_n_num = concatenated_no_n.shape[1]
t_n_num = concatenated_with_n.shape[1]
sub_car_cnt = 0  # 十字路口微观图车辆计数器
sub_car_t_cnt = 0  # 三岔路口微观图车辆计数器


# 在宏观图应用中添加路由
@app_macro.get("/")
async def root_macro():
    return {"message": "宏观图应用已连接"}


# 在微观图应用中添加路由
@app_micro.get("/")
async def root_micro():
    return {"message": "微观图应用已连接"}


# 启动宏观图应用的函数
def start_macro():
    uvicorn.run(app_macro, host="127.0.0.1", port=8080)


# 启动微观图应用的函数
def start_micro():
    uvicorn.run(app_micro, host="127.0.0.1", port=8081)


# # 获得微观图路口数量
# @app.put('/put_path_num')
# async def put_path_num(get_params: Request):  # 要在url中写参数而不是请求体
#     path_num = 4  # 预设岔路的数量
#     params = get_params.query_params
#     path_num = params.get('path_num')
#     if path_num and path_num.isdigit():  # 判断能否转换成整数
#         path_num = int(path_num)  # 将字符串转换为整数
#     else:
#         path_num = -1  # 表示转换失败
#     print(path_num)
#     if path_num == -1:
#         return {"need int"}
#     else:
#         return {"put succeed"}
#
#
# @app.get("/get_sub_num")
# async def get_sub_num():
#     num_dict = {
#         "x": no_n_num,
#         "y": n_num
#     }
#     return num_dict
#
#
# @app.get("/get_sub_path")
# async def get_sub_path():
#     no_n_list = sub_path_json(concatenated_no_n)
#     n_list = sub_path_json(concatenated_with_n)
#     global sub_car_cnt
#     switch_mode = 0  # 0是红灯前,1是红灯后
#     if sub_car_cnt >= no_n_num and switch_mode == 0:
#         switch_mode = 1  # 切换模式
#         sub_car_cnt = 0
#     if sub_car_cnt >= n_num and switch_mode == 1:
#         switch_mode = 0
#         sub_car_cnt = 0
#     print("sub_car:")
#     print("cnt/len")
#     if switch_mode == 0:
#         print(sub_car_cnt + 1, "/", no_n_num)
#         sub_car_json = no_n_list[sub_car_cnt]
#     else:
#         print(sub_car_cnt + 1, "/", n_num)
#         sub_car_json = n_list[sub_car_cnt]
#     return sub_car_json
#
#
# @app.get("/get_sub_t_path")
# async def get_sub_t_path():
#     t_no_n_list = sub_path_json(t_concatenated_no_n)
#     t_n_list = sub_path_json(t_concatenated_with_n)
#     global sub_car_t_cnt
#     switch_mode = 0  # 0是红灯前,1是红灯后
#     if sub_car_t_cnt >= t_no_n_num and switch_mode == 0:
#         switch_mode = 1  # 切换模式
#         sub_car_t_cnt = 0
#     if sub_car_t_cnt >= t_n_num and switch_mode == 1:
#         switch_mode = 0
#         sub_car_t_cnt = 0
#     print("sub_car_t:")
#     print("cnt/len")
#     if switch_mode == 0:
#         print(sub_car_t_cnt + 1, "/", t_no_n_num)
#         t_sub_car_json = t_no_n_list[sub_car_t_cnt]
#     else:
#         print(sub_car_t_cnt + 1, "/", t_n_num)
#         t_sub_car_json = t_n_list[sub_car_t_cnt]
#     return t_sub_car_json


# 主监听函数
if __name__ == "__main__":
    p1 = Process(target=start_macro)
    p2 = Process(target=start_micro)

    p1.start()
    p2.start()

    p1.join()
    p2.join()
