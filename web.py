from fastapi import FastAPI, File, UploadFile
from fastapi import Request
import uvicorn
import json
# 自定函数
from jicheng_fun import run_simulation
from jicheng_fun import G
from add_json import cars_to_json
from add_json import cars_to_file

app = FastAPI()
car_cnt = 0  # 车辆计数器
cars = []  # 全局列表cars


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
        cars = run_simulation(G=G, total_cars=car_num, round_num=1, speed=0.5)  # 直接计算path然后存成字典列表
        # 存成文件方便检查
        cars_to_file(cars)
        return {"put succeed"}


# 获取车辆路径路由
@app.get("/get_path")
async def get_path():  # 要在body中写参数
    global car_cnt
    car_list = cars_to_json(cars, add_z=-3)
    car_data = car_list[car_cnt]
    print(car_cnt)
    print(len(car_list))
    if car_cnt < len(car_list) - 1:
        car_data = car_list[car_cnt]
        car_cnt += 1
        return car_data  # 直接返回字典
    else:
        car_cnt -= len(car_list)  # 超出就减回1
        car_data = car_list[car_cnt]
        car_cnt += 1
        return car_data


# 主监听函数
if __name__ == "__main__":
    # uvicorn.run(app="web:app", host="192.168.0.92", port=8080, reload=False)
    uvicorn.run(app="web:app", host="127.0.0.1", port=8080, reload=False)
