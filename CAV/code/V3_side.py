import numpy as np
import time

def get_data( side, path ):
    with open( path, 'r' ) as file:
        info = file.readlines()

    n = len(info) - 1
    if n == 0 or n == -1:
        print( "文件为空！" )
        return

    xL, vL, xM, vM, xR, vR, rL = [], [], [], [], [], [], []
    L = M = R = 0
    rr = int(info[0])
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

def create_k(n, x, x_leader, A ):
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
    A = adjustA( A, x, n, dd )
    return A, dd, k

def create_vehicles(direction, x, v, r_side, rL, n, side ):
    given_vel = 27.0
    if direction == 'hor':
        x_leader = np.array( [float( np.min( x[:, 0] ) if side == '-' else np.max( x[:, 0] ) ), r_side] )
        # vL = np.array( [ float( round( np.mean( v[:, 0] ), 1 ) ), 0.0 ] )
        # 自定义领导者速度
        vL = np.array( [ -1 * given_vel if side == '-' else given_vel, 0.0 ] )
    elif direction == 'ver':
        x_leader = np.array( [ r_side, float( np.min( x[:, 1] ) if side == '-' else np.max( x[:, 1] ) ) ] )
        # vL = np.array( [ 0.0, float( round( np.mean( v[:, 1] ), 1) ) ] )
        vL = np.array( [ 0.0, -1 * given_vel if side == '-' else given_vel ] )

    A, x, v, rLeader = createA( n, x, v, rL, side, direction )
    return x_leader, x, v, vL, rLeader, A

def check_convergence(cnt, ts, t, flagla, flaglo, xp, posV, vp, velV, turn, r):
    if turn == 'M':
        x, y = 1, 0
    else:
        x, y = 0, 1
    if cnt == 500 or ts == t - 1:
        if flagla == 0 and np.max(np.abs(xp[:, x] - posV[-500][:, x])) < 0.01 and np.max(
                np.abs(vp[:, x] - velV[-500][:, x])) < 0.02:
            flagla = 1
        if flaglo == 0 and abs(r - np.mean(np.abs(np.diff(xp[:, y])))) < 0.01 and np.max(
                np.abs(vp[:, y] - velV[-500][:, y])) < 0.02:
            flaglo = 1
        cnt = 0
    else:
        cnt += 1
    return cnt, flagla, flaglo


def update_data(k, n, xL, x, vL, v, b, g, a, t, A, r, rL, turn, r_turn, side, status, road, direction, right_turn, keep, round ):
    stay = []
    start_time = time.time()
    # if (side == '-' and direction == 'hor') or (side == '+' and direction == 'ver'):
    #     last = -1
    # elif (side == '+' and direction == 'hor') or (side == '-' and direction == 'ver'):
    #     last = 0
    if direction == 'ver':
        idx = 1
    elif direction == 'hor':
        idx = 0
    flaglo = flagla = cnt = 0
    posV, velV, posL = [ x.copy() ], [ v.copy() ], [ xL.copy() ]     # 用于记录车辆位置更新、速度更新、领导者的位置更新

    xp, vp, lp = x.copy(), v.copy(), xL.copy()                       # 用于放车辆位置、速度、领导者的位置，先存入车辆初始位置
    R = np.zeros( ( n, n, 2 ) )                                      # 用车辆与领导者之间的理想距离计算车辆之间的相对理想距离
    for i in range(n):
        for j in range(n):
            R[i][j] = rL[i] - rL[j]
    threshold = 0.5
    light = 0
    target_time = 4.39

    for ts in range(t):
        now_time = time.time()
        # if posV[-1][3][0] <= 565.0 and posV[-1][3][0] >= 564.5:
        #     print(ts)
        # 上下
        if now_time - start_time >= target_time and keep == 0 and round == 1:
            end_time = time.time()
            print(end_time - start_time)
        # 左右
        # if ts == 20172 and keep == 0 and round == 1:
            light = 1
        if side == '-':
            min = np.argmin(posV[-1][:, idx])
            if np.all( np.abs( posV[-1][min] - r_turn ) < threshold ):
                break
        else:
            max = np.argmax(posV[-1][:, idx])
            if np.all( np.abs( posV[-1][max] - r_turn ) < threshold ):
                break
        dot_v = np.zeros_like( vp )
        for i in range(n):
            s = xp[i] - lp - rL[i]
            for j in range(n):
                # if (not ( flagla == 1 and flaglo == 1 )):
                if i != j:                                       # 当相比较的智能体不是自己时，对应的a不为0，两智能体间的关系参与调整考虑
                    if keep == 0 or round == 2:
                        dot_v[i] -= A[i][j] * (xp[i] - xp[j] - R[i][j] + b * (vp[i] - vp[j]))
                    else:
                        dot_v = np.zeros_like( vp )
                # print( vp[i], vL )
                dot_v[i] -= k[i] * (s + g * (vp[i] - vL))            # 与智能体相关联时，与领导者之间的关系参与调整考虑

        status_list = {
            'UpLeft': [75.0, '>'],
            'LeftDown': [565.0, '<'],
            'RightUp': [625.0, '>'],
            'DownRight': [15.0, '<']
        }
        stop = status_list[ status ][0]
        hypen = status_list[ status ][1]

        if light == 1 and keep == 0 and right_turn == 0:
            if direction == 'ver':
                mask = (xp[ :, 1 ] < stop if hypen == '<' else xp[ :, 1 ] > stop) & (np.abs(xp[:, 0] - road) <= 0.2)
            else:
                mask = (xp[ :, 0 ] < stop if hypen == '<' else xp[ :, 0 ] > stop) & (np.abs(xp[:, 1] - road) <= 0.2)
            print( mask )
            keep = 1
            mask = mask.astype( bool )
            vp[mask] = 0
            # print( vp )
            light_stay = posV[-1][mask]
            stay.append( light_stay )

        idx = 1 if direction == 'ver' else 0
        not_zero = vp[:, idx] != 0
        if round == 1:
            vp[not_zero] += a * dot_v[not_zero]
        elif round == 2:
            vp += a * dot_v  # 更新车辆速度位置与领导者位置
        xp += a * vp
        lp += a * vL
        # 检查是否收敛
        cnt, flagla, flaglo = check_convergence(cnt, ts, t, flagla, flaglo, xp, posV, vp, velV, turn, r)

        posV.append( xp.copy() )
        velV.append( vp.copy() )
        posL.append( lp.copy() )

    return np.array( posV ), np.array( velV ), np.array( posL ), keep, np.array( stay )









def one_car( turning_point, arriving_point, starting_direction, rL, r, n, side, status, right_turn, b, g, a, t, road, x, v, turn, round ):
    keep = 1 if round == 2 else 0
    x = x.reshape(-1, 2)
    v = v.reshape(-1, 2)
    x_leader, x, v, vL, rLeader, A = create_vehicles( starting_direction, x, v, road, rL, n, side )
    A, dd, k = create_k(n, x, x_leader, A)

    rLt = rL.copy()
    rLt[:, [0, 1]] = rLt[:, [1, 0]]

    if round == 1:
        posV, velV, posL, keep, stay = update_data(k, n, x_leader, x, vL, v, b, g, a, t, A, r, rL, 'M', turning_point, side, status, road, starting_direction, right_turn = right_turn, keep = keep, round = round )
        print( 'stay::::::::::', stay )
        x = posV[-1]
        v = velV[-1]

    x_leader = np.array(turning_point)
    if (turn == 'R' and starting_direction == 'ver') or (turn == 'L' and starting_direction == 'hor'):
        vL[0], vL[1] = vL[1], vL[0]
        rL = rLt
    elif (turn == 'L' and starting_direction == 'ver') or (turn == 'R' and starting_direction == 'hor'):
        vL[0], vL[1] = vL[1], vL[0]
        vL = -vL
        rL = -rLt
        right_turn = 1

    # x_leader = np.array(arriving_point)
    nposV, nvelV, nLpos, nkeep, nstay = update_data(k, n, x_leader, x, vL, v, b, g, a, t, A, r, rL, turn, arriving_point, side, status, road, starting_direction, right_turn = right_turn, keep = keep, round = round )
    if round == 1:
        posV = np.concatenate((posV, nposV), axis=0)
    else:
        posV = nposV
        stay = nstay
    # print( posV[-1] )
    return posV, stay





def run( side, path, info, i, single, round, stay ):
    # side -> 车道口   single -> L/M/R
    b = 1
    g = 1
    a = 0.001
    tt = 80
    t = int( tt / a )
    r_left, r_middle, r_right, r_turn, r_gap, starting_direction, status = info
    if round == 1:
        L, M, R, xL, vL, xM, vM, xR, vR, rL, r = get_data( side, path )
        # 道路信息 -> 道路中心线坐标 & 路口位置
        if single == 'M':
            n = M
            right_turn = 0
            road = r_middle
            x, v = xM, vM
        elif single == 'L':
            n = L
            right_turn = 0
            road = r_left
            x, v = xL, vL
        elif single == 'R':
            n = R
            right_turn = 1
            road = r_right
            x, v = xR, vR
    elif round == 2:
        tmp = len( stay )
        if single ==  'L':
            right_turn = 0
            road = r_left
        elif single == 'M':
            right_turn = 0
            road = r_middle
        else:
            right_turn = 1
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
    turn = single
    turning = {
        '1M': [ [600.0, 30.0], [1200.0, 30.0] ],
        '2M': [ [600.0, 60.0], [0.0, 60.0] ],
        '3M': [ [580.0, 30.0], [580.0, -570.0] ],
        '4M': [ [610.0, 60.0], [610.0, 660.0] ],
        '1L': [ [610.0, 40.0], [610.0, 640.0] ],
        '2L': [ [580.0, 50.0], [580.0, -550.0] ],
        '3L': [ [590.0, 40.0], [1190.0, 40.0] ],
        '4L': [ [600.0, 50.0], [0.0, 50.0] ],
        '1R': [ [570.0, 20.0], [570.0, -580.0] ],
        '2R': [ [620.0, 70.0], [620.0, 670.0] ],
        '3R': [ [570.0, 70.0], [-30.0, 70.0] ],
        '4R': [ [620.0, 20.0], [1220.0, 20.0] ]
    }
    # print( rL )
    if starting_direction == 'ver':
        rL = rL.copy()
        rL[:, [0, 1]] = rL[:, [1, 0]]
    # print( i, 'here')
    select = str( i ) + single
    turning_point = turning[ select ][0]
    arriving_point = turning[ select ][1]
    posV, stay = one_car( turning_point, arriving_point, starting_direction, rL, r, n, side, status, right_turn, b, g, a, t, road, x, v, turn, round )
    stay = np.array( stay )
    stay = stay.reshape(-1, 2)
    print( posV[-1], 'last of this' )
    return posV, n, stay

