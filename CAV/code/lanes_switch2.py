import numpy as np
import matplotlib.pyplot as plt
from MAS_Function import Algorithm_1
from MAS_Function import Algorithm_2
from Tests import two_steady


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


def update_data(k, n, xL, x, vL, v, b, g, a, t, A, r, rL, turn, r_turn, flag):
    flaglo = 0
    flagla = 0
    cnt = 0
    posV = [x.copy()]  # 用于记录车辆位置更新
    velV = [v.copy()]  # 用于记录车辆速度更新
    posL = [xL.copy()]  # 用于放领导者位置，先存入领导者初始位置
    xp = x.copy()  # 用于放车辆位置，先存入车辆初始位置
    vp = v.copy()  # 用于放车辆速度，先存入车辆初始速度
    lp = xL.copy()
    R = np.zeros((n, n, 2))  # 用车辆与领导者之间的理想距离计算车辆之间的相对理想距离
    for i in range(n):
        for j in range(n):
            R[i][j] = rL[i] - rL[j]
    threshold = 1.0
    for ts in range(t):
        if np.all(np.abs(posV[-1][0] - r_turn) < threshold):
            print(posV[-1][0])
            break
        dot_v = np.zeros_like(vp)  # 领导者速度不变，所有加速度为0
        for i in range(n):
            s = xp[i] - lp - rL[i]
            for j in range(n):
                if i != j:  # 当相比较的智能体不是自己时，对应的a不为0，两智能体间的关系参与调整考虑
                    dot_v[i] -= A[i][j] * (xp[i] - xp[j] - R[i][j] + b * (vp[i] - vp[j]))
            dot_v[i] -= k[i] * (s + g * (vp[i] - vL))  # 与智能体相关联时，与领导者之间的关系参与调整考虑

        vp += a * dot_v  # 更新车辆速度位置与领导者位置
        xp += a * vp

        # MAS ##############################################################################
        if ts % 400 == 0:  # and flag == 1:
            xp_mas = xp.copy()
            max_mas = np.max(xp_mas[:, 0])
            y_mas = xp_mas[:, 1]
            y_mas = np.append(y_mas, lp[1])
            x_mas = []
            for ddr in range(len(y_mas)):
                x_mas.append(max_mas)
            x_mas = np.array(x_mas)
            y_mas = np.array(y_mas)
            x_B, y_B = Algorithm_2(x_mas, y_mas, 0.2)
            xp[:, 1] = y_B[:-1] * 0.4 + xp[:, 1] * 0.6
        ####################################################################################

        lp += a * vL
        # 检查是否收敛
        cnt, flagla, flaglo = check_convergence(cnt, ts, t, flagla, flaglo, xp, posV, vp, velV, turn, r)

        posV.append(xp.copy())
        velV.append(vp.copy())
        posL.append(lp.copy())

    posV = np.array(posV)
    velV = np.array(velV)
    posL = np.array(posL)
    return posV, velV, posL, t


def create_vehicles(x, v, r_side, rL, n, r, side ):
    given_vel = 35.0
    x_leader = np.array([float(np.min(x[:, 0]) if side == 2 else np.max(x[:, 0])), r_side])
    vL = np.array([-1 * given_vel if side == 2 else given_vel, 0.0])
    A, x, v, rLeader = createA(n, x, v, rL)
    return x_leader, x, v, vL, rLeader, A


def create_k(n, x, x_leader, A):
    dd = np.mean(np.diff(x[:, 1]))
    k = np.zeros((n, 1))
    if dd != 0:
        k[0] = abs(np.round((x[0, 1] - x_leader[1]) / dd, 1))
        if n > 1:
            k[1] = abs(np.round((x[1, 1] - x_leader[1]) / dd, 1))
    else:
        k[0] = 1
        if n > 1:
            k[1] = 1
        dd = 1
    A = adjustA(A, x, n, dd)
    return A, dd, k


# def add_noise(data, epsilon, sensitivity=1.0):
#     noise = np.random.laplace(0, sensitivity/epsilon, data.shape)
#     return data + noise


def run_three2two( data, num, r_left, r_middle, r_right, ending_line ):
    b = 1
    g = 1
    a = 0.001
    tt = 50
    t = int(tt / a)
    L, M, R, xL, vL, xM, vM, xR, vR, rL, r = two_steady(data['Car_Num'], num)
    print(L, M, R)
    print(xL, '\n', xM, '\n', xR)
    # L, M, R, xL, vL, xM, vM, xR, vR, rL, r = get_data()
    # 道路信息 -> 道路中心线坐标 & 路口位置
    # r_left = 30.0
    # r_middle = 20.0
    # r_right = 10.0



    if num == 1:
        r_turn_after = [127.0, -12.5]
        r_turn_before = np.array([
            [127.0, r_left],
            [ending_line, r_middle],
            [ending_line, r_right]
        ])
    else:
        r_turn_after = [0.0, -5.0]
        r_turn_before = np.array([
            [0.0, r_left],
            [ending_line, r_middle],
            [ending_line, r_right]
        ])

    xLe, xMe, xRe = 1, 1, 1
    if xL.size != 0:
        xLe = 0
        xL_leader, xL, vL, vLL, rLeaderL, AL = create_vehicles(xL, vL, r_left, rL, L, r, num)
        AL, ddL, kL = create_k(L, xL, xL_leader, AL)
        LposV, LvelV, LposL, Lnt = update_data(kL, L, xL_leader, xL, vLL, vL, b, g, a, t, AL, r, rL, 'M', r_turn_before[0], 0)
    else:
        LposV = []

    if xM.size != 0:
        xMe = 0
        xM_leader, xM, vM, vLM, rLeaderM, AM = create_vehicles(xM, vM, r_middle, rL, M, r, num)
        AM, ddM, kM = create_k(M, xM, xM_leader, AM)
        MposV, MvelV, MposL, Mnt = update_data(kM, M, xM_leader, xM, vLM, vM, b, g, a, t, AM, r, rL, 'M', r_turn_before[1], 0)
    else:
        MposV = []

    if xR.size != 0:
        xRe = 0
        xR_leader, xR, vR, vLR, rLeaderR, AR = create_vehicles(xR, vR, r_right, rL, R, r, num)
        AR, ddR, kR = create_k(R, xR, xR_leader, AR)
        RposV, RvelV, RposL, Rnt = update_data(kR, R, xR_leader, xR, vLR, vR, b, g, a, t, AR, r, rL, 'M', r_turn_before[2], 0)
    else:
        RposV = []

    # 创建车辆信息
    # xL_leader, xL, vL, vLL, rLeaderL, AL = create_vehicles(xL, vL, r_left, rL, L, r, num)
    # xM_leader, xM, vM, vLM, rLeaderM, AM = create_vehicles(xM, vM, r_middle, rL, M, r, num)
    # xR_leader, xR, vR, vLR, rLeaderR, AR = create_vehicles(xR, vR, r_right, rL, R, r, num)

    # AL, ddL, kL = create_k(L, xL, xL_leader, AL)
    # AM, ddM, kM = create_k(M, xM, xM_leader, AM)
    # AR, ddR, kR = create_k(R, xR, xR_leader, AR)

    # 更新数据
    # LposV, LvelV, LposL, Lnt = update_data(kL, L, xL_leader, xL, vLL, vL, b, g, a, t, AL, r, rL, 'M', r_turn_before[0], 0)
    # MposV, MvelV, MposL, Mnt = update_data(kM, M, xM_leader, xM, vLM, vM, b, g, a, t, AM, r, rL, 'M', r_turn_before[1], 0)
    # RposV, RvelV, RposL, Rnt = update_data(kR, R, xR_leader, xR, vLR, vR, b, g, a, t, AR, r, rL, 'M', r_turn_before[2], 0)

    # # 为位置和速度数据添加噪声
    # LposV_noisy = add_noise(LposV, epsilon=8.5)
    # LvelV_noisy = add_noise(LvelV, epsilon=8.5)
    # MposV_noisy = add_noise(MposV, epsilon=8.5)
    # MvelV_noisy = add_noise(MvelV, epsilon=8.5)
    # RposV_noisy = add_noise(RposV, epsilon=8.5)
    # RvelV_noisy = add_noise(RvelV, epsilon=8.5)

    # 将车辆合并后更新数据
    Comb_e = 1
    if MposV[-1].size != 0 and RposV[-1].size != 0:
        Comb_e = 0
        pos_merged = np.concatenate((MposV[-1], RposV[-1]))
        vel_merged = np.concatenate((MvelV[-1], RvelV[-1]))
        n_merged = len(pos_merged)
        x = pos_merged
        v = vel_merged
        A2, x, v, rL = createA(n_merged, x, v, rL)
        if num == 1:
            x_leader = [ 40.0, -12.5 ]
        else:
            x_leader = [ 87.0, -5.0 ]
        A2, ddL, kL = create_k(n_merged, x, x_leader, A2)
        nposV, nvelV, nposL, nnt = update_data(kL, n_merged, x_leader, x, vLR, v, b, g, a, t, A2, r, rL, 'M', r_turn_after, 1)

    # # 显示图片
    # plt.figure(figsize=(10, 6))
    # for i in range(L):
    #     plt.plot(LposV[:, i, 0], LposV[:, i, 1], label=f'Vehicle {i + 1}')
    #     plt.scatter(LposV[::5000, i, 0], LposV[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
    # for i in range(M):
    #     plt.plot(MposV[:, i, 0], MposV[:, i, 1], label=f'Vehicle {i + 1}')
    #     plt.scatter(MposV[::5000, i, 0], MposV[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
    # for i in range(R):
    #     plt.plot(RposV[:, i, 0], RposV[:, i, 1], label=f'Vehicle {i + 1}')
    #     plt.scatter(RposV[::5000, i, 0], RposV[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
    # for i in range(n_merged):
    #     plt.plot(nposV[:, i, 0], nposV[:, i, 1], label=f'Vehicle {i + 1}')
    #     plt.scatter(nposV[::5000, i, 0], nposV[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置

    # plt.xlabel('X Position(m)')
    # plt.ylabel('Y Position(m)')
    # plt.show()

    return L, M, R, LposV, MposV, RposV, nposV, xLe, xMe, xRe, Comb_e
