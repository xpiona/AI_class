import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1, 1])
    b = 1.5
    tmp = np.sum(x*w) -b
    if tmp < 0.1:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1, 1])
    b = 1.5
    tmp = np.sum(x*w) -b
    if tmp < 0.1:
        return 1
    else:
        return 0


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1, 1])
    b = 1.5
    tmp = np.sum(x*w) - b
    if tmp > -0.6:
        return 1
    else:
        return 0


def XOR(x1, x2):
    o1 = OR(x1, x2)
    o2 = NAND(x1, x2)
    o3 = AND(o1, o2)

    return o3


if __name__=='__main__':  # 시작점, import로 가져올 땐 실행하지 않음#
    for xs in  [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = XOR(xs[0], xs[1])
        print(str(xs) + "->" + str(y))
