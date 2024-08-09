import matplotlib.pyplot as plt
from V2_Tjunc import run



def left2right( i, single ):
    info_left2right = [ 20.0, 10.0, 'hor', '1' ]
    path_left2right = '../data/test7.txt'
    side_left2right = '+'
    return run( side_left2right, path_left2right, info_left2right, single )

def right2left( i, single ):
    info_right2left = [ 30.0, 40.0, 'hor', '2' ]
    path_right2left = '../data/test8.txt'
    side_right2left = '-'
    return run( side_right2left, path_right2left, info_right2left, single )


def down2up( i, single ):
    info_down2up = [ 630.0, 640.0, 'ver', '3' ]
    path_down2up = '../data/test9.txt'
    side_down2up = '+'
    return run( side_down2up, path_down2up, info_down2up, single )



def draw( posV, n ):
    for i in range(n):
        plt.plot(posV[:, i, 0], posV[:, i, 1], label=f'Vehicle {i + 1}')
        print( len( posV ) )
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
            posV, n = down2up( i, single[k] )
            draw( posV, n )

    plt.xlabel('X Position(m)')
    plt.ylabel('Y Position(m)')
    plt.legend()
    plt.show()






def main():

    # stage1
    # split = [ 1, 2 ]
    # single = ['L', 'R']

    # stage2
    split = [2]
    single = ['L']




    # right_turn
    # split = [ 1, 3 ]
    # single = [ 'R', 'R' ]


    read( split, single )


if __name__ == '__main__':
    main()