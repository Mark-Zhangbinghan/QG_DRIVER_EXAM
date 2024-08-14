import numpy as np
import matplotlib.pyplot as plt
from V3_side_2 import run


def method( i, single, round, stay ):
    # 1 -> left2right
    # 2 -> right2left
    # 3 -> up2down
    # 4 -> down2up
    info_list = [
        [6.25, 3.75, 1.25, 577.5, 2.5, 'hor', 'LeftDown'],
        [8.75, 11.25, 13.75, 577.5, 2.5, 'hor', 'RightUp'],
        [575.0, 572.5, 570.0, 6.25, 2.5, 'ver', 'UpLeft' ],
        [577.5, 580.0, 582.5, 8.75, 2.5, 'ver', 'DownRight']
    ]
    path_list = [
        '../data/test4.txt',
        '../data/test2.txt',
        '../data/test5.txt',
        '../data/test6.txt'
    ]
    side_list = [
        '+', '-', '-', '+'
    ]
    return run(side_list[i-1], path_list[i-1], info_list[i-1], i, single, round, stay )

'''
def left2right( i, single ):
    info_left2right = [ 40.0, 30.0, 20.0, 600.0, 10.0, 'hor', 'LeftDown' ]
    path_left2right = '../data/test4.txt'
    side_left2right = '+'
    return run( side_left2right, path_left2right, info_left2right, i, single )

def right2left( i, single ):
    info_right2left = [ 50.0, 60.0, 70.0, 600.0, 10.0, 'hor', 'RightUp' ]
    path_right2left = '../data/test2.txt'
    side_right2left = '-'
    return run( side_right2left, path_right2left, info_right2left, i, single )


def up2down( i, single ):
    info_up2down = [ 590.0, 580.0, 570.0, 40.0, 10.0, 'ver', 'UpLeft' ]
    path_up2down = '../data/test5.txt'
    side_up2down = '-'
    return run( side_up2down, path_up2down, info_up2down, i, single )


def down2up( i, single ):
    info_down2up = [ 600.0, 610.0, 620.0, 50.0, 10.0, 'ver', 'DownRight' ]
    path_down2up = '../data/test6.txt'
    side_down2up = '+'
    return run( side_down2up, path_down2up, info_down2up, i, single )
'''


# def draw( posV, n ):
#     for i in range(n):
#         plt.plot(posV[:, i, 0], posV[:, i, 1], label=f'Vehicle {i + 1}')
#         plt.scatter(posV[::5000, i, 0], posV[::5000, i, 1], marker='<')  # 每5000个点显示一次各个车辆的位置


def read( split, single, round, stay ):
    plt.figure(figsize=(10, 6))
    ret_list = []
    posV_list1= []
    posV_list2= []
    for k, i in enumerate( split ):
        if round == 2:
            if len( stay ) == 0:
                continue
            if stay and stay[k].size == 0:
                continue
            print( 'round', i )
            posV, n, ret_stay = method( int( i ), single[k], round, stay[k] )
        elif round == 1:
            posV, n, ret_stay = method(int(i), single[k], round, stay )
        # draw(posV, int(n))
        if k == 0:
            posV_list1 = posV
        elif k == 1:
            posV_list2 = posV
        # posV -> 轨迹点
        ret_list.append( ret_stay )
    plt.xlabel('X Position(m)')
    plt.ylabel('Y Position(m)')
    plt.legend()
    plt.show()
    return ret_list, posV_list1, posV_list2






def main():
    round = 1
    stay = []
    # stage1
    split = [ 3, 4 ]
    single = ['M', 'M']
    list1, posV11, posV12 = read( split, single, round, stay )


    # stage2
    split = [1, 2]
    single = ['M', 'M']
    list2, posV21, posV22 = read( split, single, round, stay )


    # stage3
    split = [4]
    single = ['L']
    list3, posV31, posV32 = read( split, single, round, stay )


    # stage4
    split = [3]
    single = ['L']
    list4, posV41, posV42 = read( split, single, round, stay )



    # stage5
    split = [1]
    single = ['L']
    list5, posV51, posV52 = read( split, single, round, stay )


    # stage6
    split = [2]
    single = ['L']
    list6, posV61, posV62 = read( split, single, round, stay )

    # # right_turn
    # split = [ 1, 2 ]
    # single = [ 'R', 'R' ]
    # list0, posV01, posV02 = read( split, single, round, stay )
    # split = [ 3, 4 ]
    # single = [ 'R', 'R' ]
    # list0, posV03, posV04 = read( split, single, round, stay )


    ###################################################################################


    round = 2
    # stage1
    split = [3, 4]
    single = ['M', 'M']
    nlist1, posV11n, posV12n = read( split, single, round, list1 )

    # stage2
    split = [1, 2]
    single = ['M', 'M']
    nlist2, posV21n, posV22n = read( split, single, round, list2 )

    # stage3
    split = [4]
    single = ['L']
    nlist3, posV31n, posV32n = read( split, single, round, list3 )


    # stage4
    split = [3]
    single = ['L']
    nlist4, posV41n, posV42n = read( split, single, round, list4 )

    # stage5
    split = [1]
    single = ['L']
    nlist5, posV51n, posV52n = read( split, single, round, list5 )


    # stage6
    split = [2]
    single = ['L']
    nlist6, posV61n, posV62n = read( split, single, round, list6 )

    return posV11, posV12, posV21, posV22, posV31, posV41, posV51, posV61, posV11n, posV12n, posV21n, posV22n, posV31n, posV41n, posV51n, posV61n



if __name__ == '__main__':
    main()