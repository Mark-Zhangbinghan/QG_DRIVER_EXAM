import numpy as np
import matplotlib.pyplot as plt
from V2_Tjunc_2 import run


def method( i, single, round, stay ):
    # 1 -> left2right
    # 2 -> right2left
    # 3 -> up2down
    # 4 -> down2up
    info_list = [
        [ 3.75, 1.25, 'hor', '1' ],
        [ 6.25, 8.75, 'hor', '2' ],
        [ 6.25, 8.75, 'ver', '3' ]
    ]
    path_list = [
        '../data/test7.txt',
        '../data/test8.txt',
        '../data/test9.txt'
    ]
    side_list = [
        '+', '-', '+'
    ]
    return run(side_list[i-1], path_list[i-1], info_list[i-1], i, single, round, stay )




def draw( posV, n ):
    for i in range(n):
        plt.plot(posV[:, i, 0], posV[:, i, 1], label=f'Vehicle {i + 1}')
        print( len( posV ) )
        plt.scatter(posV[::5000, i, 0], posV[::5000, i, 1], marker='<')  # 每5000个点显示一次各个车辆的位置


def read( split, single, round, stay ):
    plt.figure(figsize=(10, 6))
    ret_list = []
    posV_list1 = []
    posV_list2 = []
    for k, i in enumerate( split ):
        if round == 2:
            if len( stay ) == 0:
                continue
            if stay and stay[k].size == 0:
                continue
            print( 'round', i )
            posV, n, ret_stay = method( int( i ), single[k], round, stay[k] )
            # print( stay, 'check1' )
        elif round == 1:
            posV, n, ret_stay = method(int(i), single[k], round, stay )
        draw( posV, int( n ) )
        if k == 0:
            posV_list1 = posV
        elif k == 1:
            posV_list2 = posV
        ret_list.append( ret_stay )
        # print( ret_list, 'check2')
    plt.xlabel('X Position(m)')
    plt.ylabel('Y Position(m)')
    plt.legend()
    plt.show()
    return ret_list, posV_list1, posV_list2




def main():
    round = 1
    stay = []
    # stage1
    split = [ 1, 2 ]
    single = ['L', 'R']
    list1, posV11, posV12 = read( split, single, round, stay )

    # stage2
    split = [2]
    single = ['L']
    list2, posV21, posV22 = read(split, single, round, stay)

    # stage3
    split = [ 3 ]
    single = [ 'L' ]
    list3, posV31, posV32 = read( split, single, round, stay )

    # # right_turn
    # split = [ 1, 3 ]
    # single = [ 'R', 'R' ]
    # list0, posV01, posV02 = read( split, single, round, stay )
    ####################################################################################
    round = 2

    # stage1
    split = [1, 2]
    single = ['L', 'R']
    nlist1, posV11n, posV12n = read( split, single, round, list1 )

    # stage2
    split = [2]
    single = ['L']
    nlist2, posV21n, posV22n = read( split, single, round, list2 )

    # stage3
    split = [ 3 ]
    single = [ 'L' ]
    nlist3, posV31n, posV32n = read( split, single, round, list3 )

    return [posV11, posV12, posV21, posV31, posV11n, posV12n, posV21n, posV31n]

if __name__ == '__main__':
    main()