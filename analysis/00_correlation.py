import numpy as np

def y0(x0, x1, x2, x3, x4):
    return x4&x1 ^ x3 ^ x2&x1 ^ x2 ^ x1&x0 ^ x1 ^ x0

def y1(x0, x1, x2, x3, x4):
    return x4 ^ x3&x2 ^ x3&x1 ^ x3 ^ x2&x1 ^ x2 ^ x1 ^ x0

def y2(x0, x1, x2, x3, x4):
    return x4&x3 ^ x4 ^ x2 ^ x1 ^ 1

def y3(x0, x1, x2, x3, x4):
    return x4&x0 ^ x4 ^ x3&x0 ^ x3 ^ x2 ^ x1 ^ x0

def y4(x0, x1, x2, x3, x4):
    return x4&x1 ^ x4 ^ x3 ^ x1&x0 ^ x1

def process_y4(I):
    s = []
    for k0 in range(2):
        print(f"{k0}: ", end="")
        v = []
        for n in range(4):
            n0 = (n >> 1) & 1
            n1 = n & 1
            v.append(y4(I, k0, None, n0, n1))
        print(v)
        s.append(v)

    for i in range(2):
        print(f"{i}: ", end="")
        for j in range(2):
            cor = np.corrcoef(s[i], s[j])[0,1]
            print(f"{np.abs(cor):.02f}  ", end="")
        print() 

def process(y, I = 0):
    s = []
    for k in range(4):
        k0 = (k >> 1) & 1
        k1 = k & 1
        print(f"{k0,k1}: ", end="")
        v = []
        for n in range(4):
            n0 = (n >> 1) & 1
            n1 = n & 1
            v.append(y(I, k0, k1, n0, n1))
        print(v)
        s.append(v)
    
    for i in range(4):
        print(f"{i}: ", end="")
        for j in range(4):
            cor = np.corrcoef(s[i], s[j])[0,1]
            print(f"{np.abs(cor):.02f}  ", end="")
        print()

def hw_sbox(I=0):
    s = []
    for k in range(4):
        k0 = (k >> 1) & 1
        k1 = k & 1
        print(f"{k0,k1}: ", end="")
        v = []
        for n in range(4):
            n0 = (n >> 1) & 1
            n1 = n & 1
            b0 = y0(I, k0, k1, n0, n1)
            b1 = y1(I, k0, k1, n0, n1)
            b2 = y2(I, k0, k1, n0, n1)
            b3 = y3(I, k0, k1, n0, n1)
            b4 = y4(I, k0, k1, n0, n1)
            v.append(b0+b1+b2+b3+b4)
        print(v)
        s.append(v)

    for i in range(4):
        print(f"{i}: ", end="")
        for j in range(4):
            cor = np.corrcoef(s[i], s[j])[0,1]
            print(f"{np.abs(cor):.02f}  ", end="")
        print()

def fty0(x1, x3, x4):
    return x1&(x4^1) ^ x3

def fty1(x2, x3, x4):
    return x3&(x2^1) ^ x4

def ftyprocess():
    s = []
    for k0 in range(2):
        print(f"{k0}: ", end="")
        v = []
        for n in range(4):
            n0 = (n >> 1) & 1
            n1 = n & 1
            v.append(fty0(k0, n0, n1))
        print(v)
        s.append(v)
    print(f"Correlation: {np.corrcoef(s[0], s[1])[0,1]}")

    s = []
    for k1 in range(2):
        print(f"{k1}: ", end="")
        v = []
        for n in range(4):
            n0 = (n >> 1) & 1
            n1 = n & 1
            v.append(fty1(k1, n0, n1))
        print(v)
        s.append(v)
    print(f"Correlation: {np.corrcoef(s[0], s[1])[0,1]}")

def ftzprocess():
    print("ftz0: ")
    s = []
    for k in range(8):
        print(f"{k}: ", end="")
        k00 = (k >> 2) & 1
        k36 = (k >> 1) & 1
        k45 =  k & 1
        v = []
        for n in range(64):
            n0_00 = (n >> 5) & 1
            n0_36 = (n >> 4) & 1
            n0_45 = (n >> 3) & 1
            n1_00 = (n >> 2) & 1
            n1_36 = (n >> 1) & 1
            n1_45 = (n     ) & 1
            t = fty0(k00, n0_00, n1_00) ^ fty0(k36, n0_36, n1_36) ^ fty0(k45, n0_45, n1_45)
            v.append(t)
        print(v)
        s.append(v)
    
    for i in range(8):
        print(f"{i}: ", end="")
        for j in range(8):
            cor = np.corrcoef(s[i], s[j])[0,1]
            print(f"{np.abs(cor):.02f}  ", end="")
        print()

    print("ftz1: ")
    s = []
    for k in range(8):
        print(f"{k}: ", end="")
        k00 = (k >> 2) & 1
        k03 = (k >> 1) & 1
        k25 =  k & 1
        v = []
        for n in range(64):
            n0_00 = (n >> 5) & 1
            n0_03 = (n >> 4) & 1
            n0_25 = (n >> 3) & 1
            n1_00 = (n >> 2) & 1
            n1_03 = (n >> 1) & 1
            n1_25 = (n     ) & 1
            t = fty1(k00, n0_00, n1_00) ^ fty1(k03, n0_03, n1_03) ^ fty1(k25, n0_25, n1_25)
            v.append(t)
        print(v)
        s.append(v)
    
    for i in range(8):
        print(f"{i}: ", end="")
        for j in range(8):
            cor = np.corrcoef(s[i], s[j])[0,1]
            print(f"{np.abs(cor):.02f}  ", end="")
        print()

if __name__ == "__main__":
    print("Process y0, I=0:")
    process(y0, I=0)
    print()
    print("Process y0, I=1:")
    process(y0, I=1)
    print()
    print("Process y1, I=0:")
    process(y1, I=0)
    print()
    print("Process y1, I=1:")
    process(y1, I=1)
    print()
    print("Process y2, I=0:")
    process(y2, I=0)
    print()
    print("Process y2, I=1:")
    process(y2, I=1)
    print()
    print("Process y3, I=0:")
    process(y3, I=0)
    print()
    print("Process y3, I=1:")
    process(y3, I=1)
    print()
    print("Process y4, I=0:")
    process_y4(I=0)
    print()
    print("Process y4, I=1:")
    process_y4(I=1)
    print()

    print("HW, I=0:")
    hw_sbox(I=0)
    print()
    print("HW, I=1:")
    hw_sbox(I=1)
    print()
    
    print("Fine-tuning y:")
    ftyprocess()
    print()

    print("Fine-tuning z:")
    ftzprocess()
    print()
