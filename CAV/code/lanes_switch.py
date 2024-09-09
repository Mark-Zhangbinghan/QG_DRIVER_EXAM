import numpy as np
import matplotlib.pyplot as plt
import random
from CAV.code.MAS_Function import Algorithm_1
from CAV.code.MAS_Function import Algorithm_2


# 三岔路口和十字路口的车辆不用变道
def steady_road(L, M, R, l_road, m_road, r_road, pos, width):
    xL = []
    vL = []
    xM = []
    vM = []
    xR = []
    vR = []

    for i in range(L):
        xL.append(l_road[:])
        vL.append(0)
        # vL.append( [27.0, 0.0])
        l_road[pos] += width

    for i in range(M):
        xM.append(m_road[:])
        vM.append(0)
        # vM.append([27.0, 0.0])
        m_road[pos] += width

    for i in range(R):
        xR.append(r_road[:])
        vR.append(0)
        # vR.append([27.0, 0.0])
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
        pos = 1
        width = -8
        l_road = [8.75, -20]
        m_road = [11.25, -20]
        r_road = [13.75, -20]

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
        width = -8
        l_road = [-20, 6.25]
        m_road = [-20, 3.75]
        r_road = [-20, 1.25]

    xL, vL, xM, vM, xR, vR = steady_road(L, M, R, l_road, m_road, r_road, pos, width)

    for i in range(car_num):
        r += 1
        rL.append([-1 * r, 0.0])

    # 返回最终的分配结果
    return L, M, R, xL, vL, xM, vM, xR, vR, rL, r


def get_data():
    r = 0
    name = '../data/test0.txt'
    with open(name, 'r') as file:
        info = file.readlines()
    n = len(info) - 1
    if n == 0 or n == -1:
        print("文件为空！")
        return
    xL = []
    vL = []
    xM = []
    vM = []
    xR = []
    vR = []
    rL = []
    L, M, R = 0, 0, 0
    flag = 0
    for line in info:
        if flag == 0:
            rr = float(line)  # 记录车辆的数量
            flag = 1
        else:
            data = line.split()
            if data[4] == 'L':
                xp = [float(data[0]), float(data[1])]
                vp = [float(data[2]), float(data[3])]
                xL.append(xp)
                vL.append(vp)
                L += 1
            elif data[4] == 'M':
                xp = [float(data[0]), float(data[1])]
                vp = [float(data[2]), float(data[3])]
                xM.append(xp)
                vM.append(vp)
                M += 1
            elif data[4] == 'R':
                xp = [float(data[0]), float(data[1])]
                vp = [float(data[2]), float(data[3])]
                xR.append(xp)
                vR.append(vp)
                R += 1
    for i in range(n):
        if i == 0:
            rL.append([0.0, 0.0])
        r += rr
        rL.append([-1 * r, 0.0])

    xL = np.array(xL)
    vL = np.array(vL)
    xM = np.array(xM)
    vM = np.array(vM)
    xR = np.array(xR)
    vR = np.array(vR)
    rL = np.array(rL)
    r = rr
    return L, M, R, xL, vL, xM, vM, xR, vR, rL, r


def createA(n, x, v, rL):
    A = np.ones((n, n))
    if n > 3:
        for i in range(n):
            for j in range(n):
                if j >= 3 and j - 3 - i >= 0:
                    A[i][j] = 0
                if i >= 3 and i - 3 - j >= 0:
                    A[i][j] = 0
        srt = np.argsort(x[:, 0])[::-1]
        x = x[srt]
        v = v[srt]
    rsrt = np.argsort(rL[:, 0])[::-1]
    rL = rL[rsrt]
    return A, x, v, rL


def adjustA(A, x, n, dd):
    for i in range(n):
        for j in range(n):
            if A[i][j] != 0 and i != j:
                A[i][j] = abs((x[i, 1] - x[j, 1]) / dd)
    return A



def check_convergence( cnt, ts, t, flagla, flaglo, xp, posV, vp, velV, turn, r ):
    if turn == 'M':
        x, y = 1, 0
    else:
        x, y = 0, 1
    if cnt == 500 or ts == t - 1:
        if flagla == 0 and np.max(np.abs(xp[:, x] - posV[-500][:, x])) < 0.02 and np.max(
                np.abs(vp[:, x] - velV[-500][:, x])) < 0.02:
            flagla = 1
        if flaglo == 0 and abs(r - np.mean(np.abs(np.diff(xp[:, y])))) < 0.05 and np.max(
                np.abs(vp[:, y] - velV[-500][:, y])) < 0.02:
            flaglo = 1
        cnt = 0
    else:
        cnt += 1
    return cnt, flagla, flaglo


def update_data( k, n, xL, x, vL, v, b, g, a, t, A, r, rL, turn, r_turn, flag ):
    flaglo = 0
    flagla = 0
    cnt = 0
    posV = [x.copy()]               # 用于记录车辆位置更新
    velV = [v.copy()]               # 用于记录车辆速度更新
    posL = [ xL.copy()]             # 用于放领导者位置，先存入领导者初始位置
    xp = x.copy()                   # 用于放车辆位置，先存入车辆初始位置
    vp = v.copy()                   # 用于放车辆速度，先存入车辆初始速度
    lp = xL.copy()
    R = np.zeros( ( n, n, 2 ) )     # 用车辆与领导者之间的理想距离计算车辆之间的相对理想距离
    for i in range(n):
        for j in range(n):
            R[i][j] = rL[i] - rL[j]
    threshold = 1.0
    for ts in range( t ):
        if np.all( np.abs( posV[-1][0] - r_turn ) < threshold ):
            print( posV[-1][0] )
            break
        dot_v = np.zeros_like( vp )                                 # 领导者速度不变，所有加速度为0
        for i in range( n ):
            s = xp[i] - lp - rL[i]
            for j in range( n ):
                if i != j:  # 当相比较的智能体不是自己时，对应的a不为0，两智能体间的关系参与调整考虑
                    dot_v[i] -= A[i][j] * (xp[i] - xp[j] - R[i][j] + b * (vp[i] - vp[j]))
            dot_v[i] -= k[i] * (s + g * (vp[i] - vL))  # 与智能体相关联时，与领导者之间的关系参与调整考虑

        vp += a * dot_v  # 更新车辆速度位置与领导者位置
        xp += a * vp

        if ts % 400 == 0: # and flag == 1:
            xp_mas = xp.copy()
            max_mas = np.max( xp_mas[:, 0] )
            y_mas = xp_mas[:, 1]
            y_mas = np.append( y_mas, lp[1] )
            x_mas = []
            for ddr in range( len( y_mas ) ):
                x_mas.append( max_mas )
            x_mas = np.array(x_mas)
            y_mas = np.array(y_mas)
            x_B, y_B = Algorithm_2(x_mas, y_mas, 0.2)
            xp[:, 1] = y_B[:-1] * 0.4 + xp[:, 1] * 0.6




        lp += a * vL
        # 检查是否收敛
        cnt, flagla, flaglo = check_convergence( cnt, ts, t, flagla, flaglo, xp, posV, vp, velV, turn, r )

        posV.append(xp.copy())
        velV.append(vp.copy())
        posL.append(lp.copy())

    posV = np.array(posV)
    velV = np.array(velV)
    posL = np.array(posL)
    return posV, velV, posL, t




def create_vehicles( side, x, v, r_side, rL, n, r ):
    given_vel = 25.0
    x_leader = np.array( [float( np.min( x[:, 0] ) if side == 2 else np.max( x[:, 0] ) ), r_side] )
    vL = np.array( [ -1 * given_vel if side == 2 else given_vel, 0.0 ] )
    A, x, v, rLeader = createA( n, x, v, rL )
    return x_leader, x, v, vL, rLeader, A


def create_k( n, x, x_leader, A ):
    dd = np.mean( np.diff( x[ :, 1 ] ) )
    k = np.zeros( ( n, 1 ) )
    if dd != 0:
        k[0] = abs(np.round((x[0, 1] - x_leader[1]) / dd, 1))
        if n > 1:
            k[1] = abs(np.round((x[1, 1] - x_leader[1]) / dd, 1))
    else:
        k[0] = 1
        if n > 1:
            k[1] = 1
        dd = 1
    A = adjustA( A, x, n, dd )
    return A, dd, k


def run( data, num, r_left, r_middle, r_right, ending_line ):
    b = 1
    g = 1
    a = 0.001
    tt = 40
    t = int(tt / a)
    # L, M, R, xL, vL, xM, vM, xR, vR, rL, r = get_data()
    # data = {'PathNum': 3, 'Car_Num': 3}
    L, M, R, xL, vL, xM, vM, xR, vR, rL, r = two(data['Car_Num'], num)
    print(L, M, R)
    print(xL, xM, xR)
    print( rL)
    # 道路信息 -> 道路中心线坐标 & 路口位置
    # r_left = -10.0
    # r_middle = -12.5
    # r_right = -15.0
    # ending_line = 127.0
    r_turn_before = np.array([
        [ending_line, r_left],
        [ending_line, r_middle],
        [ending_line, r_right]
    ])

    # r_turn_after = [800.0, 20.0]
    xLe, xMe, xRe = 1, 1, 1
    if xL.size != 0:
        xLe = 0
        xL_leader, xL, vL, vLL, rLeaderL, AL = create_vehicles( num, xL, vL, r_left, rL, L, r )
        AL, ddL, kL = create_k( L, xL, xL_leader, AL )
        print( xL_leader, vLL)
        LposV, LvelV, LposL, Lnt = update_data( kL, L, xL_leader, xL, vLL, vL, b, g, a, t, AL, r, rL, 'M', r_turn_before[0], 0 )
    else:
        LposV = []

    if xM.size != 0:
        xMe = 0
        xM_leader, xM, vM, vLM, rLeaderM, AM = create_vehicles( num, xM, vM, r_middle, rL, M, r )
        AM, ddM, kM = create_k( M, xM, xM_leader, AM )
        MposV, MvelV, MposL, Mnt = update_data( kM, M, xM_leader, xM, vLM, vM, b, g, a, t, AM, r, rL, 'M', r_turn_before[1], 0 )
    else:
        MposV = []


    if xR.size != 0:
        xRe = 0
        xR_leader, xR, vR, vLR, rLeaderR, AR = create_vehicles( num, xR, vR, r_right, rL, R, r )
        AR, ddR, kR = create_k( R, xR, xR_leader, AR)
        RposV, RvelV, RposL, Rnt = update_data( kR, R, xR_leader, xR, vLR, vR, b, g, a, t, AR, r, rL, 'M', r_turn_before[2], 0 )
    else:
        RposV = []
    # rLt = rL.copy()
    # rLt[ :, [0, 1] ] = rLt[ :, [1, 0] ]

    #
    # pos_merged = np.concatenate( ( MposV[-1], RposV[-1] ) )
    # vel_merged = np.concatenate( ( MvelV[-1], RvelV[-1] ) )
    # n_merged = len( pos_merged )
    # x = pos_merged
    # v = vel_merged
    # A2, x, v, rL = createA( n_merged, x, v, rL )
    # x_leader = [400.0, 20.0]
    # A2, ddL, kL = create_k( n_merged, x, x_leader, A2 )
    # nposV, nvelV, nposL, nnt = update_data( kL, n_merged, x_leader, x, vLR, v, b, g, a, t, A2, r, rL, 'M', r_turn_after, 1 )


    return L, M, R, LposV, MposV, RposV, xLe, xMe, xRe




    # # 显示图片
    # plt.figure(figsize=(10, 6))
    # if xL.size != 0:
    #     for i in range(L):
    #         plt.plot(LposV[:, i, 0], LposV[:, i, 1], label=f'Vehicle {i + 1}')
    #         plt.scatter(LposV[::5000, i, 0], LposV[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
    #         print(len(LposV))
    #
    # if xM.size != 0:
    #     for i in range(M):
    #         plt.plot(MposV[:, i, 0], MposV[:, i, 1], label=f'Vehicle {i + 1}')
    #         plt.scatter(MposV[::5000, i, 0], MposV[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
    #         print( len(MposV) )
    #
    # if xR.size != 0:
    #     for i in range(R):
    #         plt.plot(RposV[:, i, 0], RposV[:, i, 1], label=f'Vehicle {i + 1}')
    #         plt.scatter(RposV[::5000, i, 0], RposV[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
    #         print( len(RposV))
    #
    #
    # # for i in range( n_merged ):
    # #     plt.plot(nposV[:, i, 0], nposV[:, i, 1], label=f'Vehicle {i + 1}')
    # #     plt.scatter(nposV[::5000, i, 0], nposV[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
    # # for i in range(R2):
    # #     plt.plot(nRposV[:, i, 0], nRposV[:, i, 1], label=f'Vehicle {i + 1}')
    # #     plt.scatter(nRposV[::5000, i, 0], nRposV[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
    #
    #
    #
    # plt.xlabel('X Position(m)')
    # plt.ylabel('Y Position(m)')
    # # plt.legend()
    # plt.show()



