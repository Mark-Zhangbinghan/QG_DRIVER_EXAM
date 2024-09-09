import json
import numpy as np
from lanes_switch import run


def get_data(json_data):
    num = 1  # 设置用来判断为道路的哪一条路
    # 使用json.loads()函数将JSON字符串解析为Python字典
    data = json.loads(json_data)
    pathnum = data['PathNum']
    car_num = data['Car_Num']

    # 获取车辆数量
    if pathnum == 2:

        data1 = {'PathNum': 3, 'Car_Num': car_num[0]}
        left2right = [-10.0, -12.5, -15.0, 127.0]
        L, M, R, LposV, MposV, RposV, xLe, xMe, xRe = run(data1, 1, left2right[0], left2right[1], left2right[2],
                                                          left2right[3])

        data2 = {'PathNum': 3, 'Car_Num': car_num[1]}
        right2left = [-7.5, -5.0, -2.5, -5.0]
        L2, M2, R2, LposV2, MposV2, RposV2, xLe2, xMe2, xRe2 = run(data2, 2, right2left[0], right2left[1],
                                                                   right2left[2], right2left[3])
    else:
        car_num = data['CarNum']
