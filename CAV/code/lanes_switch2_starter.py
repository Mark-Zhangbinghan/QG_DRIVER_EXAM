from matplotlib import pyplot as plt
from lanes_switch2 import run_three2two

data1 = {'PathNum': 3, 'Car_Num': 6}
left2right = [ -10.0, -12.5, -15.0, 20.0 ]
L, M, R, LposV, MposV, RposV, nposV, xLe, xMe, xRe, ce = run_three2two( data1, 1, left2right[0], left2right[1], left2right[2], left2right[3] )

data2 = {'PathNum': 3, 'Car_Num': 7}
right2left = [ -7.5, -5.0, -2.5, 107.0 ]
L2, M2, R2, LposV2, MposV2, RposV2, nposV2, xLe2, xMe2, xRe2, ce2 = run_three2two( data2, 2, right2left[0], right2left[1], right2left[2], right2left[3] )

plt.figure(figsize=(10, 6))
if xLe == 0:
    for i in range(L):
        plt.plot(LposV[:, i, 0], LposV[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(LposV[::5000, i, 0], LposV[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
        print(len(LposV))

if xMe == 0:
    for i in range(M):
        plt.plot(MposV[:, i, 0], MposV[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(MposV[::5000, i, 0], MposV[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
        print( len(MposV) )

if xRe == 0:
    for i in range(R):
        plt.plot(RposV[:, i, 0], RposV[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(RposV[::5000, i, 0], RposV[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
        print( len(RposV))

if ce == 0:
    for i in range(R2):
        plt.plot(nposV[:, i, 0], nposV[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(nposV[::5000, i, 0], nposV[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
        print(len(RposV))

if xLe == 0:
    for i in range(L2):
        plt.plot(LposV2[:, i, 0], LposV2[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(LposV2[::5000, i, 0], LposV2[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
        print(len(LposV2))

if xMe2 == 0:
    for i in range(M2):
        plt.plot(MposV2[:, i, 0], MposV2[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(MposV2[::5000, i, 0], MposV2[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
        print(len(MposV2))

if xRe2 == 0:
    for i in range(R2):
        plt.plot(RposV2[:, i, 0], RposV2[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(RposV2[::5000, i, 0], RposV2[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
        print(len(RposV2))

if ce2 == 0:
    for i in range(R2):
        plt.plot(nposV2[:, i, 0], nposV2[:, i, 1], label=f'Vehicle {i + 1}')
        plt.scatter(nposV2[::5000, i, 0], nposV2[::5000, i, 1], marker='>')  # 每5000个点显示一次各个车辆的位置
        print(len(nposV2))



plt.xlabel('X Position(m)')
plt.ylabel('Y Position(m)')
# plt.legend()
plt.show()