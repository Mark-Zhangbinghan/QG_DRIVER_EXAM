from CAV.code.lanes_switch2 import run_three2two
from CAV.code.starter import main
from CAV.code.starter_Tjunc import t_main


def get_data(json_data):
    num = 1  # 设置用来判断为道路的哪一条路
    # 使用json.loads()函数将JSON字符串解析为Python字典
    data = json_data
    pathnum = data['PathNum']
    car_num = data['Car_Num']
    car_num[1] = car_num[0] / 2
    car_num[0] = car_num[0] - car_num[1]
    if pathnum == 2:

        data1 = {'PathNum': 3, 'Car_Num': car_num[0]}
        left2right = [ -10.0, -12.5, -15.0, 20.0 ]
        L, M, R, LposV, MposV, RposV, xLe, xMe, xRe = run_three2two(data1, 1, left2right[0], left2right[1], left2right[2],
                                                          left2right[3])

        data2 = {'PathNum': 3, 'Car_Num': car_num[1]}
        right2left = [ -7.5, -5.0, -2.5, 107.0 ]
        L2, M2, R2, LposV2, MposV2, RposV2, xLe2, xMe2, xRe2 = run_three2two(data2, 2, right2left[0], right2left[1],
                                                                   right2left[2], right2left[3])
        return [LposV, MposV, RposV, LposV2, MposV2, RposV2]

    if pathnum == 3:
        path_list = t_main(car_num)
        return path_list

    elif pathnum == 4:
        path_list = main(car_num)
        return path_list


# if __name__ == '__main__':
#     json_data = {
#         'PathNum': 2,
#         'Car_Num': [4, 5]
#     }
#     get_data(json_data)
