import numpy as np
import matplotlib.pyplot as plt
from V3_side import run


def method( i, single, round, stay ):
    # 1 -> left2right
    # 2 -> right2left
    # 3 -> up2down
    # 4 -> down2up
    info_list = [
        [40.0, 30.0, 20.0, 600.0, 10.0, 'hor', 'LeftDown'],
        [50.0, 60.0, 70.0, 600.0, 10.0, 'hor', 'RightUp'],
        [590.0, 580.0, 570.0, 40.0, 10.0, 'ver', 'UpLeft' ],
        [600.0, 610.0, 620.0, 50.0, 10.0, 'ver', 'DownRight']
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


def draw( posV, n ):
    for i in range(n):
        plt.plot(posV[:, i, 0], posV[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(posV[::5000, i, 0], posV[::5000, i, 1], marker='<')  # 每5000个点显示一次各个车辆的位置


def read( split, single, round, stay ):
    plt.figure(figsize=(10, 6))
    ret_list = []
    for k, i in enumerate( split ):
        if round == 2:
            if len( stay ) == 0:
                continue
            if stay and stay[k].size == 0:
                continue
            print( 'round', i )
            posV, n, ret_stay = method( int( i ), single[k], round, stay[k] )
            print( stay, 'check1' )
        elif round == 1:
            posV, n, ret_stay = method(int(i), single[k], round, stay )
        # posV -> 轨迹点
        draw( posV, int( n ) )

        ret_list.append( ret_stay )
        print( ret_list, 'check2')
    plt.xlabel('X Position(m)')
    plt.ylabel('Y Position(m)')
    plt.legend()
    plt.show()
    return ret_list





def main():
    round = 1
    stay = []
    # # stage1
    # split = [ 3, 4 ]
    # single = ['M', 'M']
    # list1 = read( split, single, round, stay )
    # # print( '#################', list1 )

    # stage2
    split = [1, 2]
    single = ['M', 'M']
    list2 = read( split, single, round, stay )

    # # stage3
    # split = [4]
    # single = ['L']
    # list3 = read( split, single, round, stay )
    #
    # # stage4
    # split = [3]
    # single = ['L']
    # list4 = read( split, single, round, stay )
    #
    # # stage5
    # split = [1]
    # single = ['L']
    # list5 = read( split, single, round, stay )
    #
    #
    # # stage6
    # split = [2]
    # single = ['L']
    # list6 = read( split, single, round, stay )
    #
    #
    # # right_turn
    # split = [ 1, 2, 3, 4 ]
    # single = [ 'R', 'R', 'R', 'R' ]
    # list0 = read( split, single, round, stay )

    ###################################################################################
    round = 2

    # # stage1
    # split = [3, 4]
    # single = ['M', 'M']
    # nlist1 = read( split, single, round, list1 )

    # # stage2
    split = [1, 2]
    single = ['M', 'M']
    nlist2 = read( split, single, round, list2 )

    # # stage3
    # split = [4]
    # single = ['L']
    # nlist3 = read( split, single, round, list3 )
    #
    # # stage4
    # split = [3]
    # single = ['L']
    # nlist4 = read( split, single, round, list4 )
    #
    # # stage5
    # split = [1]
    # single = ['L']
    # nlist5 = read( split, single, round, list5 )
    #
    # # stage6
    # split = [2]
    # single = ['L']
    # nlist6 = read( split, single, round, list6 )



if __name__ == '__main__':
    main()