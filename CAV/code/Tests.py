import json
import random
import numpy as np


# 三岔路口和十字路口的车辆不用变道
def steady_road(L, M, R, l_road, m_road, r_road, pos, width, num):
    xL = []
    vL = []
    xM = []
    vM = []
    xR = []
    vR = []
    speed_1 = [ 5.0, 0.0 ]
    speed_2 = [-5.0, 0.0]
    speed_3 = [0.0, -5.0]
    speed_4 = [0.0, 5.0]
    # print( "###########", num )
    for i in range(L):
        xL.append(l_road[:])
        # vL.append(0)
        if num == 1:
            vL.append( speed_1 )
        elif num == 2:
            vL.append( speed_2 )
        elif num == 3:
            vL.append( speed_3 )
        elif num == 4:
            vL.append( speed_4 )
        l_road[pos] += width

    for i in range(M):
        xM.append(m_road[:])
        # vM.append(0)
        if num == 1:
            vM.append(speed_1)
        elif num == 2:
            vM.append(speed_2)
        elif num == 3:
            vM.append(speed_3)
        elif num == 4:
            vM.append(speed_4)
        m_road[pos] += width

    for i in range(R):
        xR.append(r_road[:])
        # vR.append(0)
        if num == 1:
            vR.append(speed_1)
        elif num == 2:
            vR.append(speed_2)
        elif num == 3:
            vR.append(speed_3)
        elif num == 4:
            vR.append(speed_4)
        r_road[pos] += width

    return xL, vL, xM, vM, xR, vR


# 给直道上的车辆随机分配起始道路和最后所在的道路
def random_road(car_num, L, M, R, l_road, m_road, r_road, pos, width, num):
    xL = []
    vL = []
    xM = []
    vM = []
    xR = []
    vR = []

    # 初始化车辆数量列表
    remaining_cars = [L, M, R]
    roads = [l_road, m_road, r_road]
    if num == 1:
        side = 1
    else:
        side = -1

    if car_num < 4:
        road_index = random.randint(0, 2)
        road_index2 = random.randint(0, 2)
        while road_index == road_index2:
            road_index2 = random.randint(0, 2)
        for i in range( car_num ):
            road = roads[road_index]
            if road_index2 == 0:
                xL.append(road[:])
                """vL.append(0)"""
                vL.append([side * 27.0, 0.0])
                L, M, R = car_num, 0, 0
            elif road_index2 == 1:
                xM.append(road[:])
                """vM.append(0)"""
                vM.append([side * 27.0, 0.0])
                L, M, R = 0, car_num, 0
            else:
                xR.append(road[:])
                """vR.append(0)"""
                vR.append([side * 27.0, 0.0])
                L, M, R = 0, 0, car_num
            road[pos] += width
    else:
        for i in range(car_num):
            if sum(remaining_cars) == 0:
                break  # 如果没有剩余车辆，退出循环

            # 随机选择一条道路
            road_index = random.randint(0, 2)

            while remaining_cars[road_index] == 0:
                road_index = random.randint(0, 2)  # 确保选择的道路有剩余车辆
            # print( road_index )
            road = roads[road_index]
            if i < L:
                xL.append(road[:])
                """vL.append(0)"""
                vL.append( [side * 27.0, 0.0] )
            elif i < (L + M):
                xM.append(road[:])
                """vM.append(0)"""
                vM.append([side * 27.0, 0.0])
            else:
                xR.append(road[:])
                """vR.append(0)"""
                vR.append([side * 27.0, 0.0])
            road[pos] += width
            remaining_cars[road_index] -= 1
    return xL, vL, xM, vM, xR, vR, L, M, R


def two(car_num, num):
    if car_num == 1:
        car_num = 2
    if car_num == 3:
        car_num = 4
    if car_num == 5:
        car_num = 6
    # 处理两条路径的逻辑
    r = 0
    rr = 5
    rL = []
    pos = 0     # 要修改的坐标索引(0:x, 1:z)
    width = 8   # 车辆间距

    # 确保每个方向最后至少有两辆车
    L, M, R = 2, 2, 2
    remaining_cars = car_num - (L + M + R)  # 计算剩余车辆数量

    # 随机分配剩余的车辆到三个方向
    directions = [L, M, R]  # 创建一个列表来存储方向数量
    while remaining_cars > 0:
        random.shuffle(directions)  # 随机打乱顺序
        for i, direction in enumerate(directions):
            if remaining_cars > 0:
                directions[i] += 1
                remaining_cars -= 1
    L, M, R = directions[0], directions[1], directions[2]

    if num == 1:    # 第一种情况
        width = -8
        l_road = [0.0, -10.0]
        m_road = [0.0, -12.5]
        r_road = [0.0, -15.0]

    else:   # 第二种情况
        l_road = [127.5, -7.5]
        m_road = [127.5, -5.0]
        r_road = [127.5, -2.5]

    xL, vL, xM, vM, xR, vR, L, M, R = random_road(car_num, L, M, R, l_road, m_road, r_road, pos, width, num)

    side = -1
    if num == 2:
        side = 1

    for i in range(car_num):
        r += rr
        rL.append([side * r, 0.0])

    xL = np.array(xL)
    vL = np.array(vL)
    xM = np.array(xM)
    vM = np.array(vM)
    xR = np.array(xR)
    vR = np.array(vR)
    rL = np.array(rL)

    # 返回最终的分配结果
    return L, M, R, xL, vL, xM, vM, xR, vR, rL, r


def two_steady(car_num, num):
    if car_num == 1:
        car_num = 2
    if car_num == 3:
        car_num = 4
    if car_num == 5:
        car_num = 6
    # 处理两条路径的逻辑
    r = 0
    rr = 5
    rL = []
    pos = 0     # 要修改的坐标索引(0:x, 1:z)
    width = 8   # 车辆间距

    # 确保每个方向最后至少有两辆车
    L, M, R = 2, 2, 2
    remaining_cars = car_num - (L + M + R)  # 计算剩余车辆数量

    # 随机分配剩余的车辆到三个方向
    directions = [L, M, R]  # 创建一个列表来存储方向数量
    while remaining_cars > 0:
        random.shuffle(directions)  # 随机打乱顺序
        for i, direction in enumerate(directions):
            if remaining_cars > 0:
                directions[i] += 1
                remaining_cars -= 1
    L, M, R = directions[0], directions[1], directions[2]

    if num == 1:    # 第一种情况
        width = -8
        l_road = [0.0, -10.0]
        m_road = [0.0, -12.5]
        r_road = [0.0, -15.0]

    else:   # 第二种情况
        l_road = [127.5, -7.5]
        m_road = [127.5, -5.0]
        r_road = [127.5, -2.5]

    xL, vL, xM, vM, xR, vR, L, M, R = random_road2(car_num, L, M, R, l_road, m_road, r_road, pos, width, num)

    side = -1
    if num == 2:
        side = 1

    for i in range(car_num):
        r += rr
        rL.append([side * r, 0.0])

    xL = np.array(xL)
    vL = np.array(vL)
    xM = np.array(xM)
    vM = np.array(vM)
    xR = np.array(xR)
    vR = np.array(vR)
    rL = np.array(rL)

    # 返回最终的分配结果
    return L, M, R, xL, vL, xM, vM, xR, vR, rL, r
def random_road2(car_num, L, M, R, l_road, m_road, r_road, pos, width, num):
    xL = []
    vL = []
    xM = []
    vM = []
    xR = []
    vR = []

    # 初始化车辆数量列表
    remaining_cars = [L, M, R]
    roads = [l_road, m_road, r_road]
    if num == 1:
        side = 1
    else:
        side = -1

    if car_num < 4:
        road_index = random.randint(0, 2)
        # road_index2 = random.randint(0, 2)
        # while road_index == road_index2:
        #     road_index2 = random.randint(0, 2)
        for i in range( car_num ):
            road = roads[road_index]
            if road_index == 0:
                xL.append(road[:])
                """vL.append(0)"""
                vL.append([side * 27.0, 0.0])
                L, M, R = car_num, 0, 0
            elif road_index == 1:
                xM.append(road[:])
                """vM.append(0)"""
                vM.append([side * 27.0, 0.0])
                L, M, R = 0, car_num, 0
            else:
                xR.append(road[:])
                """vR.append(0)"""
                vR.append([side * 27.0, 0.0])
                L, M, R = 0, 0, car_num
            road[pos] += width
    else:
        for i in range(car_num):
            if sum(remaining_cars) == 0:
                break  # 如果没有剩余车辆，退出循环

            # 随机选择一条道路
            road_index = random.randint(0, 2)

            while remaining_cars[road_index] == 0:
                road_index = random.randint(0, 2)  # 确保选择的道路有剩余车辆
            # print( road_index )
            road = roads[road_index]
            if road_index == 0:
                xL.append(road[:])
                """vL.append(0)"""
                vL.append([side * 27.0, 0.0])
                # L, M, R = car_num, 0, 0
            elif road_index == 1:
                xM.append(road[:])
                """vM.append(0)"""
                vM.append([side * 27.0, 0.0])
                # L, M, R = 0, car_num, 0
            else:
                xR.append(road[:])
                """vR.append(0)"""
                vR.append([side * 27.0, 0.0])
                # L, M, R = 0, 0, car_num
            road[pos] += width
            remaining_cars[road_index] -= 1
    return xL, vL, xM, vM, xR, vR, L, M, R


def three(car_num, num):
    # 处理三条路径的逻辑
    rr = 5
    r = 0
    rL = []
    pos = 0  # 要修改的坐标索引(0:x, 1:z)
    width = 8  # 车辆间距
    L, M, R = 0, 0, 0

    if num == 1:    # 第一种情况
        if car_num <= 2:
            L, M = 2, 0  # 至少保证 L 组有 2 辆车
        else:
            # 随机分配剩余车辆
            remaining_cars = car_num - 2
            L = random.randint(2, remaining_cars + 2)  # L 组至少有 2 辆车，最多有剩余车辆数 + 2 辆车
            M = car_num - L
        l_road = [36, 13.75]
        m_road = [36, 11.25]
        r_road = []

    elif num == 2:   # 第二种情况
        if car_num <= 2:
            R, M = 2, 0  # 至少保证 R 组有 2 辆车
        else:
            # 随机分配剩余车辆
            remaining_cars = car_num - 2
            R = random.randint(2, remaining_cars + 2)  # R 组至少有 2 辆车，最多有剩余车辆数 + 2 辆车
            M = car_num - R
        width = -8
        l_road = []
        m_road = [-26, 16.25]
        r_road = [-26, 18.75]

    else:   # 第三种情况
        if car_num <= 4:
            L, R = 2, 2  # 至少保证 L 和 R 组有 2 辆车
        else:
            # 随机分配剩余车辆
            remaining_cars = car_num - 4
            L = random.randint(2, remaining_cars)  # L 组至少有 2 辆车，最多有剩余车辆数 + 2 辆车
            R = car_num - L
        pos = 1
        width = -8
        l_road = [6.25, -45]
        m_road = []
        r_road = [8.25, -45]

    xL, vL, xM, vM, xR, vR = steady_road(L, M, R, l_road, m_road, r_road, pos, width)

    for i in range(car_num):
        r += rr
        rL.append([-1 * r, 0.0])


    # 返回最终的分配结果
    return L, M, R, xL, vL, xM, vM, xR, vR, rL, rr


def four(car_num, num):
    # 处理四条路径的逻辑
    r = 0
    rL = []
    pos = 0  # 要修改的坐标索引(0:x, 1:z)
    width = 8  # 车辆间距

    # 确保每个方向最后至少有两辆车
    L, M, R = 2, 2, 2
    remaining_cars = car_num - (L + M + R)  # 计算剩余车辆数量

    # 随机分配剩余的车辆到三个方向
    directions = [L, M, R]  # 创建一个列表来存储方向数量
    while remaining_cars > 0:
        random.shuffle(directions)  # 随机打乱顺序
        for i, direction in enumerate(directions):
            if remaining_cars > 0:
                directions[i] += 1
                remaining_cars -= 1
    L, M, R = directions[0], directions[1], directions[2]

    if num == 1:    # 第一种情况
        width = -8
        l_road = [-20, 6.25]
        m_road = [-20, 3.75]
        r_road = [-20, 1.25]

    elif num == 2:  # 第二种情况
        l_road = [40, 8.75]
        m_road = [40, 11.25]
        r_road = [40, 13.75]

    elif num == 3:  # 第三种情况
        pos = 1
        l_road = [6.25, 40]
        m_road = [3.75, 40]
        r_road = [1.25, 40]

    else:   # 第四种情况
        pos = 1
        width = -8
        l_road = [8.75, -20]
        m_road = [11.25, -20]
        r_road = [13.75, -20]

    xL, vL, xM, vM, xR, vR = steady_road(L, M, R, l_road, m_road, r_road, pos, width)

    for i in range(car_num):
        r += 1
        rL.append([-1 * r, 0.0])

    # 返回最终的分配结果
    return L, M, R, xL, vL, xM, vM, xR, vR, rL, r


def get_data(json_data):
    num = 1  # 设置用来判断为道路的哪一条路
    # 使用json.loads()函数将JSON字符串解析为Python字典
    data = json.loads(json_data)

    # 使用字典将路径数量与对应的函数关联起来
    path_functions = {
        2: two,
        3: three,
        4: four
    }

    # 获取路径数量
    pathnum = data['PathNum']
    # 获取车辆数量
    if pathnum == 2:
        car_num = [data['CarNum']/2, data['CarNum']/2]
    else:
        car_num = data['CarNum']
    # 根据路径数量调用对应的函数
    for carnum in car_num:
        if pathnum in path_functions:
            L, M, R, xL, vL, xM, vM, xR, vR, rL, r = path_functions[pathnum](carnum, num)
            num += 1
        else:
            raise ValueError("Unsupported path number")


if __name__ == '__main__':
    data = {'PathNum': 3, 'Car_Num': 6}
    L, M, R, xL, vL, xM, vM, xR, vR, rL, r = two_steady(data['Car_Num'], 1)
    print(L, M, R )
    print( xL, '\n', xM, '\n',  xR )
    # print(rL, r)
