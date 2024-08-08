import numpy as np
import matplotlib.pyplot as plt
from V3_side import run



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



def draw( posV, n ):
    for i in range(n):
        plt.plot(posV[:, i, 0], posV[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(posV[::5000, i, 0], posV[::5000, i, 1], marker='<')  # 每5000个点显示一次各个车辆的位置


def read( split, single ):
    plt.figure(figsize=(10, 6))
    for k, i in enumerate( split ):
        i = int( i )
        if i == 1:
            posV, n = left2right( i, single[k] )
            draw( posV, n )
        elif i == 2:
            posV, n = right2left( i, single[k] )
            draw( posV, n )
        elif i == 3:
            posV, n = up2down( i,  single[k] )
            draw( posV, n )
        elif i == 4:
            posV, n = down2up( i, single[k] )
            draw( posV, n )

    plt.xlabel('X Position(m)')
    plt.ylabel('Y Position(m)')
    plt.legend()
    plt.show()



    '''
    for i in range(L):
        plt.plot(LposV[:, i, 0], LposV[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(LposV[::5000, i, 0], LposV[::5000, i, 1], marker='<')  # 每5000个点显示一次各个车辆的位置
    for i in range(M):
        plt.plot(MposV[:, i, 0], MposV[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(MposV[::5000, i, 0], MposV[::5000, i, 1], marker='<')  # 每5000个点显示一次各个车辆的位置
    for i in range(R):
        plt.plot(RposV[:, i, 0], RposV[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(RposV[::5000, i, 0], RposV[::5000, i, 1], marker='<')  # 每5000个点显示一次各个车辆的位置
    
    for i in range(L2):
        plt.plot(LposV2[:, i, 0], LposV2[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(LposV2[::5000, i, 0], LposV2[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
    for i in range(M2):
        plt.plot(MposV2[:, i, 0], MposV2[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(MposV2[::5000, i, 0], MposV2[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
    for i in range(R2):
        plt.plot(RposV2[:, i, 0], RposV2[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(RposV2[::5000, i, 0], RposV2[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
    

    for i in range(L3):
        plt.plot(LposV3[:, i, 0], LposV3[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(LposV3[::5000, i, 0], LposV3[::5000, i, 1], marker='<')  # 每5000个点显示一次各个车辆的位置
    for i in range(M3):
        plt.plot(MposV3[:, i, 0], MposV3[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(MposV3[::5000, i, 0], MposV3[::5000, i, 1], marker='<')  # 每5000个点显示一次各个车辆的位置
    for i in range(R3):
        plt.plot(RposV3[:, i, 0], RposV3[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(RposV3[::5000, i, 0], RposV3[::5000, i, 1], marker='<')  # 每5000个点显示一次各个车辆的位置
    
    for i in range(L4):
        plt.plot(LposV4[:, i, 0], LposV4[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(LposV4[::5000, i, 0], LposV4[::5000, i, 1], marker='<')  # 每5000个点显示一次各个车辆的位置
    for i in range(M4):
        plt.plot(MposV4[:, i, 0], MposV4[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(MposV4[::5000, i, 0], MposV4[::5000, i, 1], marker='<')  # 每5000个点显示一次各个车辆的位置
    for i in range(R4):
        plt.plot(RposV4[:, i, 0], RposV4[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(RposV4[::5000, i, 0], RposV4[::5000, i, 1], marker='<')  # 每5000个点显示一次各个车辆的位置
    '''





def main():
    '''
    # stage1
    split = [ 3, 4 ]
    single = ['M', 'M']

    # stage2
    split = [1, 2]
    single = ['M', 'M']


    # stage3
    split = [4]
    single = ['L']

    # stage4
    split = [3]
    single = ['L']
    '''

    # stage5
    # split = [1]
    # single = ['L']

    # stage6
    # split = [2]
    # single = ['L']


    # right_turn
    split = [ 1, 2, 3, 4 ]
    single = [ 'R', 'R', 'R', 'R' ]


    read( split, single )


if __name__ == '__main__':
    main()