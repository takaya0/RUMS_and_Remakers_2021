import numpy as np


def sym(A):
    return (A + A.T)/2





while True:
    arr = np.random.randint(-10, 31, (10, 10))
    sym_arr = sym(arr)
    max_eign = np.max(np.linalg.eigh(sym_arr)[0])
    if abs(max_eign - int(max_eign)) < 1e-5:
        print(sym_arr)
        print(max_eign)
        break






