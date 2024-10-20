import numpy as np
from CAV.code.MAS_Function import Algorithm_1
from CAV.code.MAS_Function import Algorithm_2
import random


def get_data(side, path):
    with open(path, 'r') as file:
        info = file.readlines()

    n = len(info) - 1
    if n == 0 or n == -1:
        print("文件为空！")
        return

    xL, vL, xR, vR, rL = [], [], [], [], []
    L = R = 0
    rr = len(info[0])

    for line in info[1:]:
        data = line.split()
        xp = [float(data[0]), float(data[1])]
        vp = [float(data[2]), float(data[3])]
        if data[4] == 'L':
            xL.append(xp)
            vL.append(vp)
            L += 1
        elif data[4] == 'R':
            xR.append(xp)
            vR.append(vp)
            R += 1

    if side == '-':
        ehp = 1
    else:
        ehp = -1

    for i in range(n):
        rL.append([ehp * rr * i, 0.0])

    return L, R, np.array(xL), np.array(vL), np.array(xR), np.array(vR), np.array(rL), rr


def createA(n, x, v, rL, side, direction):
    A = np.ones((n, n))
    if n > 3:
        for i in range(n):
            for j in range(n):
                if j >= 3 and j - 3 - i >= 0:
                    A[i][j] = 0
                if i >= 3 and i - 3 - j >= 0:
                    A[i][j] = 0

        if direction == 'hor':
            srt = np.argsort(x[:, 0]) if side == '-' else np.argsort(x[:, 0])[::-1]
        elif direction == 'ver':
            srt = np.argsort(x[:, 0])[::-1] if side == '-' else np.argsort(x[:, 0])
        x, v = x[srt], v[srt]
    rL = rL[np.argsort(rL[:, 0])[::-1]]
    return A, x, v, rL


def adjustA(A, x, n, dd, side):
    for i in range(n):
        for j in range(n):
            if A[i][j] != 0 and i != j:
                if side == 'hor':
                    A[i][j] = abs((x[i, 0] - x[j, 0]) / dd)
                elif side == 'ver':
                    A[i][j] = abs((x[i, 1] - x[j, 1]) / dd)
    return A

def create_k(n, x, x_leader, A, side ):
    if len( x ) == 1:
        dd = 0
    else:
        dd = np.mean( np.diff(x[:, 1]) )
    k = np.zeros( (n, 1) )
    if dd != 0:
        k[0] = abs(np.round((x[0, 1] - x_leader[1]) / dd, 1))
        if n > 1:
            k[1] = abs(np.round((x[1, 1] - x_leader[1]) / dd, 1))
    else:
        k[0] = 1
        if n > 1:
            k[1] = 1
        dd = 1
    A = adjustA( A, x, n, dd, side )
    return A, dd, k


def create_vehicles(direction, x, v, r_side, rL, n, side):
    given_vel = 10.0
    if direction == 'hor':
        x_leader = np.array([float(np.min(x[:, 0]) if side == '-' else np.max(x[:, 0])), r_side])
        # vL = np.array( [ float( round( np.mean( v[:, 0] ), 1 ) ), 0.0 ] )
        # 自定义领导者速度
        vL = np.array([-1 * given_vel if side == '-' else given_vel, 0.0])
    elif direction == 'ver':
        x_leader = np.array([r_side, float(np.min(x[:, 1]) if side == '-' else np.max(x[:, 1]))])
        # vL = np.array( [ 0.0, float( round( np.mean( v[:, 1] ), 1) ) ] )
        vL = np.array([0.0, -1 * given_vel if side == '-' else given_vel])

    A, x, v, rLeader = createA(n, x, v, rL, side, direction)
    return x_leader, x, v, vL, rLeader, A


def check_convergence(cnt, ts, t, flagla, flaglo, xp, posV, vp, velV, turn, r):
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


def update_data(k, n, xL, x, vL, v, b, g, a, t, A, r, rL, turn, r_turn, side, status, road, direction, right_turn, keep, round, stage ):
    stay = []

    if direction == 'ver':
        idx = 1
    elif direction == 'hor':
        idx = 0
    flaglo = flagla = cnt = 0
    posV, velV, posL = [x.copy()], [v.copy()], [xL.copy()]  # 用于记录车辆位置更新、速度更新、领导者的位置更新

    xp, vp, lp = x.copy(), v.copy(), xL.copy()  # 用于放车辆位置、速度、领导者的位置，先存入车辆初始位置
    R = np.zeros((n, n, 2))  # 用车辆与领导者之间的理想距离计算车辆之间的相对理想距离
    for i in range(n):
        for j in range(n):
            R[i][j] = rL[i] - rL[j]
    threshold = 0.5
    light = 1
    if stage != 0:
        target_time_s = 15000 * (stage - 1)
        target_time_e = 15000 * stage
    # print( target_time_s, target_time_e )
    for ts in range(t):
        status = str(status)
        status_list = {
            '1': [-2.0, '<'],
            '2': [12.0, '>'],
            '3': [-2.0, '<']
        }
        stop = status_list[status][0]
        hypen = status_list[status][1]
        reach = 0
        now_time = ts

        if stage != 0:  # and keep == 0
            if now_time >= target_time_s and now_time <= target_time_e:  # 可以通行
                light = 0
            elif now_time < target_time_s and now_time > target_time_e:  # 红灯
                light = 1

        if direction == 'ver':
            r_line = [road, stop]
        else:
            r_line = [stop, road]

        if side == '-':
            min = np.argmin(posV[-1][:, idx])
            # print(posV[-1][min], r_line)
            if np.all(np.abs(posV[-1][min] - r_line) < threshold):
                reach = 1
        else:
            max = np.argmax(posV[-1][:, idx])
            # print(posV[-1][max], r_line)
            if np.all(np.abs(posV[-1][max] - r_line) < threshold):
                reach = 1

        if reach == 1 and light == 1 and keep == 0:
            vp = np.zeros_like( vp )

        dot_v = np.zeros_like(vp)
        for i in range(n):
            s = xp[i] - lp - rL[i]
            for j in range(n):
                # if (not (flagla == 1 and flaglo == 1)):
                if i != j:
                    if keep == 0 or round == 2:
                        dot_v[i] -= A[i][j] * (xp[i] - xp[j] - R[i][j] + b * (vp[i] - vp[j]))
                    else:
                        dot_v = np.zeros_like(vp)
                # print( vp[i], vL )
                dot_v[i] -= k[i] * (s + g * (vp[i] - vL))  # 与智能体相关联时，与领导者之间的关系参与调整考虑


        if reach == 1 and light == 1 and keep == 0:
            if direction == 'ver':
                mask = (xp[:, 1] < stop if hypen == '<' else xp[:, 1] > stop) & (np.abs(xp[:, 0] - road) <= 0.2)
            else:
                mask = (xp[:, 0] < stop if hypen == '<' else xp[:, 0] > stop) & (np.abs(xp[:, 1] - road) <= 0.2)
            # print(mask)
            keep = 1
            mask = mask.astype(bool)
            vp[mask] = 0
            # print( vp )
            light_stay = posV[-1][mask]
            stay.append(light_stay)

        idx = 1 if direction == 'ver' else 0
        not_zero = vp[:, idx] != 0
        if round == 1:
            vp[not_zero] += a * dot_v[not_zero]
        elif round == 2:
            vp += a * dot_v  # 更新车辆速度位置与领导者位置
        xp += a * vp

        # if ts % 400 == 0:
        #     xp_mas = xp.copy()
        #     if (direction == 'ver' and round == 1) or (direction == 'hor' and round == 2 and turn != 'M'):
        #         ym, xm = 0, 1
        #     elif (direction == 'hor' and round == 1) or (direction == 'ver' and round == 2 and turn != 'M'):
        #         xm, ym = 0, 1
        #     max_mas = np.max(xp_mas[:, xm])
        #     y_mas = xp_mas[:, ym]
        #     y_mas = np.append(y_mas, lp[ym])
        #     x_mas = []
        #     for ddr in range(len(y_mas)):
        #         x_mas.append(max_mas)
        #     x_mas = np.array(x_mas)
        #     y_mas = np.array(y_mas)
        #     x_B, y_B = Algorithm_2(x_mas, y_mas, 0.2)
        #     xp[:, ym] = y_B[:-1] * 0.4 + xp[:, ym] * 0.6

        lp += a * vL
        # 检查是否收敛
        cnt, flagla, flaglo = check_convergence(cnt, ts, t, flagla, flaglo, xp, posV, vp, velV, turn, r)

        posV.append(xp.copy())
        velV.append(vp.copy())
        posL.append(lp.copy())

    return np.array(posV), np.array(velV), np.array(posL), keep, np.array(stay)


def three(car_num, num):
    if car_num == 1:
        car_num = 2
    if car_num == 3:
        car_num = 4
    if car_num == 5:
        car_num = 6
    # 处理三条路径的逻辑
    rr = 5
    r = 0
    rL = []
    pos = 0  # 要修改的坐标索引(0:x, 1:z)
    width = 8  # 车辆间距
    L, M, R = 0, 0, 0

    if num == 2:    # 第一种情况
        if car_num <= 2:
            L, M = 2, 0  # 至少保证 L 组有 2 辆车
        else:
            # 随机分配剩余车辆
            remaining_cars = car_num - 2
            L = random.randint(2, remaining_cars)  # L 组至少有 2 辆车，最多有剩余车辆数 + 2 辆车
            R = car_num - L
        r_road = [36.0, 8.75]
        l_road = [36.0, 6.25]
        m_road = []

    elif num == 1:   # 第二种情况
        if car_num <= 2:
            R, M = 2, 0  # 至少保证 R 组有 2 辆车
        else:
            # 随机分配剩余车辆
            remaining_cars = car_num - 2
            R = random.randint(2, remaining_cars)  # R 组至少有 2 辆车，最多有剩余车辆数 + 2 辆车
            L = car_num - R
        width = -8
        m_road = []
        r_road = [-36.0, 1.25]
        l_road = [-36.0, 3.75]

    else:   # 第三种情况
        if car_num <= 4:
            L, R = 2, 2  # 至少保证 L 和 R 组有 2 辆车
        else:
            # 随机分配剩余车辆
            remaining_cars = car_num - 2
            L = random.randint(2, remaining_cars)  # L 组至少有 2 辆车，最多有剩余车辆数 + 2 辆车
            R = car_num - L
        pos = 1
        width = -8
        l_road = [6.25, -45.0]
        m_road = []
        r_road = [8.25, -45.0]

    xL, vL, xM, vM, xR, vR = steady_road(L, M, R, l_road, m_road, r_road, pos, width, num)

    side = -1
    if num == 2 or num == 3:
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
    return L, M, R, xL, vL, xM, vM, xR, vR, rL, rr

def steady_road(L, M, R, l_road, m_road, r_road, pos, width, num):
    xL = []
    vL = []
    xM = []
    vM = []
    xR = []
    vR = []
    speed_1 = [ 10.0, 0.0 ]
    speed_2 = [-10.0, 0.0]
    speed_4 = [0.0, -10.0]
    speed_3 = [0.0, 10.0]
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



def one_car(turning_point, arriving_point, starting_direction, rL, r, n, side, status, right_turn, b, g, a, t, road, x,
            v, turn, round, stage ):
    keep = 1 if round == 2 else 0
    x = x.reshape(-1, 2)
    v = v.reshape(-1, 2)
    x_leader, x, v, vL, rLeader, A = create_vehicles(starting_direction, x, v, road, rL, n, side)
    A, dd, k = create_k(n, x, x_leader, A, starting_direction )

    rLt = rL.copy()
    rLt[:, [0, 1]] = rLt[:, [1, 0]]


    if round == 1:
        posV, velV, posL, keep, stay = update_data(k, n, x_leader, x, vL, v, b, g, a, t, A, r, rL, 'M', turning_point,
                                                   side, status, road, starting_direction, right_turn=right_turn,
                                                   keep=keep, round=round, stage = stage )

        x = posV[-1]
        v = velV[-1]
    if keep == 0:
        if side == '-':
            if starting_direction == 'ver':
                l_x = turning_point[0]
                l_y = np.min(x[:, 1])
            elif starting_direction == 'hor':
                l_x = np.min(x[:, 0])
                l_y = turning_point[1]
        else:
            if starting_direction == 'ver':
                l_x = turning_point[0]
                l_y = np.max(x[:, 1])
            elif starting_direction == 'hor':
                l_x = np.max(x[:, 0])
                l_y = turning_point[1]
        x_leader = np.array([l_x, l_y])
    else:
        x_leader = np.array(turning_point)

    if (turn == 'R' and starting_direction == 'ver') or (turn == 'L' and starting_direction == 'hor'):
        vL[0], vL[1] = vL[1], vL[0]
        rL = rLt
    elif (turn == 'L' and starting_direction == 'ver') or (turn == 'R' and starting_direction == 'hor'):
        vL[0], vL[1] = vL[1], vL[0]
        vL = -vL
        rL = -rLt
        right_turn = 1

    nposV, nvelV, nLpos, nkeep, nstay = update_data(k, n, x_leader, x, vL, v, b, g, a, t, A, r, rL, turn,
                                                    arriving_point, side, status, road, starting_direction,
                                                    right_turn=right_turn, keep=keep, round=round, stage = stage )
    if round == 1:
        posV = np.concatenate((posV, nposV), axis=0)
    else:
        posV = nposV
        stay = nstay
    return posV, stay


def run( side, car_n, info, i, single, round, stay, s ):
    b = 1
    g = 1
    a = 0.001
    tt = 60
    t = int(tt / a)
    r_left, r_right, starting_direction, status = info
    status = str(status)
    if round == 1:
        # L, R, xL, vL, xR, vR, rL, r = get_data(side, path)
        L, M, R, xL, vL, xM, vM, xR, vR, rL, r = three(car_n, s)
        print( L, M, R )
        print( xL, xM, xR )
        if single == 'L':
            if status == '1':
                turn = 'M'
            else:
                turn = single
            n = L
            right_turn = 0
            road = r_left
            x, v = xL, vL
        elif single == 'R':
            n = R
            if status == '2':
                turn = 'M'
                right_turn = 0
            else:
                right_turn = 1
                turn = single
            road = r_right
            x, v = xR, vR
    elif round == 2:
        tmp = len( stay )
        if single == 'L':
            if status == '1':
                turn = 'M'
            else:
                turn = single
            right_turn = 0
            road = r_left
        elif single == 'R':
            if status == '2':
                turn = 'M'
                right_turn = 0
            else:
                right_turn = 1
                turn = single
            road = r_right
        x = np.array( stay )
        n = tmp
        v = np.zeros_like( x )
        r = 10
        ehp = 1 if side == '-' else -1
        rL = []
        for k in range( tmp ):
            rL.append([ehp * r * float(k), 0.0])
        rL = np.array( rL )

    turning = {
        '1L': [[6.25, 3.75], [36.0, 3.75]],
        '1R': [[1.25, 1.25], [1.25, -45.0]],
        '2L': [[3.75, 6.25], [3.75, -36.0]],
        '2R': [[3.75, 8.75], [-36.0, 8.75]],
        '3L': [[6.25, 6.25], [-36.0, 6.25]],
        '3R': [[8.75, 1.25], [36.0, 1.25]],
    }

    stage_list = {
        '1L': 1,
        '2L': 2,
        '3L': 3,
        '1R': 4,
        '2R': 1,
        '3R': 4
    }
    if starting_direction == 'ver':
        rL = rL.copy()
        rL[:, [0, 1]] = rL[:, [1, 0]]
    select = status + single
    turning_point = turning[select][0]
    arriving_point = turning[select][1]
    stage = stage_list[ select ]
    print( turning_point, arriving_point, select )

    posV, stay = one_car(turning_point, arriving_point, starting_direction, rL, r, n, side, status, right_turn, b, g, a, t, road, x, v, turn, round, stage )
    stay = np.array( stay )
    stay = stay.reshape(-1, 2)
    return posV, n, stay
