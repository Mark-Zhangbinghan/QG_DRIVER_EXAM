import numpy as np
import matplotlib.pyplot as plt

def get_data( side, path ):
    with open( path, 'r' ) as file:
        info = file.readlines()

    n = len(info) - 1
    if n == 0 or n == -1:
        print("文件为空！")
        return

    xL, vL, xM, vM, xR, vR, rL = [], [], [], [], [], [], []
    L = M = R = 0
    rr = len( info[0] )

    for line in info[1:]:
        data = line.split()
        xp = [float(data[0]), float(data[1])]
        vp = [float(data[2]), float(data[3])]
        if data[4] == 'L':
            xL.append(xp)
            vL.append(vp)
            L += 1
        elif data[4] == 'M':
            xM.append(xp)
            vM.append(vp)
            M += 1
        elif data[4] == 'R':
            xR.append(xp)
            vR.append(vp)
            R += 1

    if side == '-':
        ehp = 1
    else:
        ehp = -1

    for i in range(n):
        rL.append( [ ehp * rr * i, 0.0 ] )

    return L, M, R, np.array(xL), np.array(vL), np.array(xM), np.array(vM), np.array(xR), np.array(vR), np.array(rL), rr


def createA(n, x, v, rL, side, direction ):
    A = np.ones((n, n))
    if n > 3:
        for i in range(n):
            for j in range(n):
                if j >= 3 and j - 3 - i >= 0:
                    A[i][j] = 0
                if i >= 3 and i - 3 - j >= 0:
                    A[i][j] = 0

        if direction == 'hor':
            srt = np.argsort( x[ :, 0 ] ) if side == '-' else np.argsort( x[ :, 0 ] )[ ::-1 ]
        elif direction == 'ver':
            srt = np.argsort( x[ :, 0 ] )[ ::-1 ] if side == '-' else np.argsort( x[ :, 0 ] )
        x, v = x[srt], v[srt]
    rL = rL[ np.argsort( rL[:, 0] )[::-1] ]
    return A, x, v, rL


def adjustA(A, x, n, dd):
    for i in range(n):
        for j in range(n):
            if A[i][j] != 0 and i != j:
                A[i][j] = abs((x[i, 1] - x[j, 1]) / dd)
    return A


def check_convergence(cnt, ts, t, flagla, flaglo, xp, posV, vp, velV, turn, a, r):
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


def update_data(k, n, xL, x, vL, v, b, g, a, t, A, r, rL, turn, r_turn, side ):
    if side == '-':
        last = -1
    else:
        last = 0
    flaglo = flagla = cnt = 0
    posV, velV, posL = [ x.copy() ], [ v.copy() ], [ xL.copy() ]     # 用于记录车辆位置更新、速度更新、领导者的位置更新

    xp, vp, lp = x.copy(), v.copy(), xL.copy()                       # 用于放车辆位置、速度、领导者的位置，先存入车辆初始位置
    R = np.zeros( ( n, n, 2 ) )                                      # 用车辆与领导者之间的理想距离计算车辆之间的相对理想距离
    for i in range(n):
        for j in range(n):
            R[i][j] = rL[i] - rL[j]
    threshold = 1.0
    for ts in range(t):
        if np.all( np.abs( posV[-1][last] - r_turn ) < threshold ):
            break
        dot_v = np.zeros_like( vp )  # 领导者速度不变，所有加速度为0
        for i in range(n):
            s = xp[i] - lp - rL[i]
            for j in range(n):
                if not ( flagla == 1 and flaglo == 1 ):
                    if i != j:  # 当相比较的智能体不是自己时，对应的a不为0，两智能体间的关系参与调整考虑
                        dot_v[i] -= A[i][j] * (xp[i] - xp[j] - R[i][j] + b * (vp[i] - vp[j]))
                dot_v[i] -= k[i] * (s + g * (vp[i] - vL))  # 与智能体相关联时，与领导者之间的关系参与调整考虑

        vp += a * dot_v  # 更新车辆速度位置与领导者位置
        xp += a * vp
        lp += a * vL
        # 检查是否收敛
        cnt, flagla, flaglo = check_convergence(cnt, ts, t, flagla, flaglo, xp, posV, vp, velV, turn, a, r)

        posV.append( xp.copy() )
        velV.append( vp.copy() )
        posL.append( lp.copy() )


    return np.array( posV ), np.array( velV ), np.array( posL )


def create_vehicles(direction, x, v, r_side, rL, n, side ):
    if direction == 'hor':
        x_leader = np.array( [float( np.min( x[:, 0] ) if side == '-' else np.max( x[:, 0] ) ), r_side] )
        vL = np.array( [ float( round( np.mean( v[:, 0] ), 1 ) ), 0.0 ] )
    elif direction == 'ver':
        x_leader = np.array( [ r_side, float( np.min( x[:, 1] ) if side == '-' else np.max( x[:, 1] ) ) ] )
        vL = np.array( [ 0.0, float( round( np.mean( v[:, 1] ), 1) ) ] )

    A, x, v, rLeader = createA( n, x, v, rL, side, direction )
    return x_leader, x, v, vL, rLeader, A


def create_k(n, x, x_leader, A ):
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
    A = adjustA( A, x, n, dd )
    return A, dd, k


def run( side, path, info ):
    b = 1
    g = 1
    a = 0.001
    tt = 100
    t = int( tt / a )
    L, M, R, xL, vL, xM, vM, xR, vR, rL, r = get_data( side, path )

    # 道路信息 -> 道路中心线坐标 & 路口位置
    r_left, r_middle, r_right, r_turn, r_gap, starting_direction, status = info
    status_set = {
        'RightUp': [ -2, 1, 2, -2, 2, 2 ],
        'LeftDown': [ 1, 1, -3, 1, 2, -3 ],
        'UpLeft': [ 0, -1, 3, 1, -1, -1 ],
        'RightDown': [ 0, 1, -3, -1, 1, 1 ]
    }
    sta = status_set[ status ]
    if starting_direction == 'hor':
        r_turn_before = np.array( [
            [ r_turn + sta[0] * r_gap, r_left ],
            [ sta[1] * r_turn, r_middle ],
            [ r_turn + sta[2] * r_gap, r_right ]
        ] )
        r_turn_after = np.array( [
            [ r_turn + sta[3] * r_gap, r_turn ],
            [ sta[4] * r_turn, r_middle ],
            [ r_turn + sta[5] * r_gap, r_right - r_turn ]
        ] )
    elif starting_direction == 'ver':
        r_turn_before = np.array([
            [ r_left, r_turn + sta[0] * r_gap ],
            [ r_middle, r_turn + sta[1] * r_gap ],
            [ r_right, r_turn + sta[2] * r_gap ]
        ])
        r_turn_after = np.array([
            [ r_left + sta[3] * r_left, r_turn ],
            [ r_middle, r_turn + sta[1] * r_gap + sta[4] * r_middle ],
            [ r_right + sta[5] * r_right, r_turn + sta[2] * r_gap ]
        ])
        rL = rL.copy()
        rL[:, [0, 1]] = rL[:, [1, 0]]


    # 创建车辆信息
    xL_leader, xL, vL, vLL, rLeaderL, AL = create_vehicles(starting_direction, xL, vL, r_left, rL, L, side )
    xM_leader, xM, vM, vLM, rLeaderM, AM = create_vehicles(starting_direction, xM, vM, r_middle, rL, M, side )
    xR_leader, xR, vR, vLR, rLeaderR, AR = create_vehicles(starting_direction, xR, vR, r_right, rL, R, side )


    AL, ddL, kL = create_k(L, xL, xL_leader, AL )
    AM, ddM, kM = create_k(M, xM, xM_leader, AM )
    AR, ddR, kR = create_k(R, xR, xR_leader, AR )

    rLt = rL.copy()
    rLt[:, [0, 1]] = rLt[:, [1, 0]]

    LposV, LvelV, LposL = update_data(kL, L, xL_leader, xL, vLL, vL, b, g, a, t, AL, r, rL, 'M', r_turn_before[0], side )
    xL_leader = LposL[-1]
    xL = LposV[-1]
    vL = LvelV[-1]
    vLL[0], vLL[1] = vLL[1], vLL[0]
    nLposV, nLvelV, nLposL = update_data(kL, L, xL_leader, xL, vLL, vL, b, g, a, t, AL, r, rLt, 'L', r_turn_after[0], side )
    # print('1')


    MposV, MvelV, MposL = update_data(kM, M, xM_leader, xM, vLM, vM, b, g, a, t, AM, r, rL, 'M', r_turn_before[1], side )
    xM_leader = MposL[-1]
    xM = MposV[-1]
    vM = MvelV[-1]
    nMposV, nMvelV, nMposL = update_data(kM, M, xM_leader, xM, vLM, vM, b, g, a, t, AM, r, rL, 'M', r_turn_after[1], side )
    # print('2')


    RposV, RvelV, RposL = update_data(kR, R, xR_leader, xR, vLR, vR, b, g, a, t, AR, r, rL, 'M', r_turn_before[2], side )
    xR_leader = RposL[-1]
    xR = RposV[-1]
    vR = RvelV[-1]
    vLR[0], vLR[1] = vLR[1], vLR[0]
    vLRt = -vLR
    rLtt = -rLt
    nRposV, nRvelV, nRposL = update_data(kR, R, xR_leader, xR, vLRt, vR, b, g, a, t, AR, r, rLtt, 'R', r_turn_after[2], side )
    # print('3')

    LposV = np.concatenate((LposV, nLposV), axis=0)
    # LvelV = np.concatenate((LvelV, nLvelV), axis=0)
    MposV = np.concatenate((MposV, nMposV), axis=0)
    # MvelV = np.concatenate((MvelV, nMvelV), axis=0)
    RposV = np.concatenate((RposV, nRposV), axis=0)
    # RvelV = np.concatenate((RvelV, nRvelV), axis=0)


    return LposV, MposV, RposV, L, M, R

def draw():
    LposV, MposV, RposV, L, M, R = run()
    plt.figure(figsize=(10, 6))
    for i in range(L):
        plt.plot(LposV[:, i, 0], LposV[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(LposV[::5000, i, 0], LposV[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
    for i in range(M):
        plt.plot(MposV[:, i, 0], MposV[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(MposV[::5000, i, 0], MposV[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
    for i in range(R):
        plt.plot(RposV[:, i, 0], RposV[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(RposV[::5000, i, 0], RposV[::5000, i, 1], marker='<')  # 每5000个点显示一次各个车辆的位置

    plt.xlabel('X Position(m)')
    plt.ylabel('Y Position(m)')
    plt.legend()
    plt.show()
