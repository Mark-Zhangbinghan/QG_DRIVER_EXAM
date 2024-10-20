import numpy as np
import matplotlib.pyplot as plt
from CAV.code.V2_Tjunc_2 import run


def method(i, single, round, stay, car_n ):
    # 1 -> left2right
    # 2 -> right2left
    # 3 -> up2down
    # 4 -> down2up
    info_list = [
        [3.75, 1.25, 'hor', '1'],
        [6.25, 8.75, 'hor', '2'],
        [6.25, 8.75, 'ver', '3']
    ]
    path_list = [
        'CAV/data/test7.txt',
        'CAV/data/test8.txt',
        'CAV/data/test9.txt'
    ]
    side_list = [
        '+', '-', '+'
    ]
    return run(side_list[i - 1], car_n[i-1], info_list[i - 1], i, single, round, stay, i )


def draw(posV, n):
    for i in range(n):
        plt.plot(posV[:, i, 0], posV[:, i, 1], label=f'Vehicle {i + 1}')
        print(len(posV))
        plt.scatter(posV[::5000, i, 0], posV[::5000, i, 1], marker='<')  # 每5000个点显示一次各个车辆的位置


def read(split, single, round, stay, car_n):
    plt.figure(figsize=(10, 6))
    ret_list = []
    posV_list1 = []
    posV_list2 = []
    for k, i in enumerate(split):
        if round == 2:
            if len(stay) == 0:
                continue
            if stay and stay[k].size == 0:
                continue
            # print('round', i)
            posV, n, ret_stay = method(int(i), single[k], round, stay[k], car_n)
        elif round == 1:
            posV, n, ret_stay = method(int(i), single[k], round, stay, car_n)
        draw(posV, int(n))
        if k == 0:
            posV_list1 = posV
        elif k == 1:
            posV_list2 = posV
        # posV -> 轨迹点
        ret_list.append(ret_stay)
    plt.xlabel('X Position(m)')
    plt.ylabel('Y Position(m)')
    plt.legend()
    plt.show()
    return ret_list, posV_list1, posV_list2


def t_main(car_num):
    round = 1
    stay = []
    car_n = car_num
    # stage1
    split = [1, 2]
    single = ['L', 'R']
    list1, posV11, posV12 = read(split, single, round, stay, car_n)

    # stage2
    split = [2]
    single = ['L']
    list2, posV21, posV22 = read(split, single, round, stay, car_n)
    #
    # stage3
    split = [3]
    single = ['L']
    list3, posV31, posV32 = read(split, single, round, stay, car_n)

    # right_turn
    split = [ 1, 3 ]
    single = [ 'R', 'R' ]
    list0, posV01, posV02 = read(split, single, round, stay, car_n)
    ####################################################################################
    round = 2

    # stage1
    split = [1, 2]
    single = ['L', 'R']
    nlist1, posV11n, posV12n = read(split, single, round, list1, car_n)
    #
    # stage2
    split = [2]
    single = ['L']
    nlist2, posV21n, posV22n = read(split, single, round, list2, car_n)
    #
    # stage3
    split = [3]
    single = ['L']
    nlist3, posV31n, posV32n = read(split, single, round, list3, car_n)

    # right_turn
    split = [1, 3]
    single = ['R', 'R']
    nlist0, posV01n, posV02n = read(split, single, round, list0, car_n)

    return [posV11, posV12, posV21, posV31, posV01, posV02, posV11n, posV12n, posV21n, posV31n, posV01n, posV02n]


if __name__ == '__main__':
    t_main()
