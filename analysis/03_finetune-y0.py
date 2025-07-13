import matplotlib.pyplot as plt
import numpy as np
import os

class MatCPA:

    def __init__(self, T):
        # Load trace matrix
        self.T = T

        # Calculate mean and sigma
        self.Tm = self._mean(self.T)
        self.Ts = self._std_dev(self.T, self.Tm)
    
    def _mean(self, X):
        return np.sum(X, axis=0)/len(X)

    def _std_dev(self, X, X_bar):
        return np.sqrt(np.sum((X-X_bar)**2, axis=0))

    def _cov(self, X, X_bar, Y, Y_bar):
        a = (X-X_bar)
        b = (Y-Y_bar)
        return np.sum(a*b, axis=0)
    
    def do_cpa(self, nn, h):
        if h.shape == (nn,): h = h.reshape((nn,1))
        hm = self._mean(h)
        hs = self._std_dev(h, hm)
        cor = self._cov(self.T, self.Tm, h, hm) / (self.Ts * hs)
        return np.abs(cor)

def load_trace_matrix(nt_s, nt_e, nn_s, nn_e, path_to_traces):
    trace_files = [os.path.join(path_to_traces, f) for f in os.listdir(path_to_traces) if ".npy" in f]
    trace_files = sorted(trace_files)

    if nn_s >= nn_e or nt_s >= nt_e: raise ValueError
    if nn_e > len(trace_files): raise ValueError
    
    T = np.zeros((nt_e-nt_s, nn_e-nn_s), dtype=np.float64)    
    for i in range(nn_e-nn_s):
        tr = np.load(trace_files[i+nn_s], mmap_mode='r')
        T[:,i] = tr[nt_s:nt_e]
    
    T = T.transpose()
    print(f"[INFO] Loaded matrix of traces: ", end="")
    print(f"shape {T.shape}, size {T.size * T.itemsize} B or {T.size * T.itemsize / (10**9):.4f} GB")
    return T

def y0(x0, x1, x2, x3, x4):
    return x4&x1 ^ x3 ^ x2&x1 ^ x2 ^ x1&x0 ^ x1 ^ x0

def fty0(x1, x3, x4):
    return x1&(x4^1) ^ x3

# def fty1(x2, x3, x4):
#     return x3&(x2^1) ^ x4

def HW(v):
    return sum([int(b) for b in bin(v)[2:]])

if __name__ == "__main__":
    nn_s = 0
    nn_e = 1000
    nt_s = 105
    nt_e = 817
    path_to_traces = "../unprotected/traces"
    path_to_nonces = "../unprotected/nonces.npy"
    T = load_trace_matrix(nt_s, nt_e, nn_s, nn_e, path_to_traces)
    # print(T[0], T.shape, T.dtype)
    N = np.load(path_to_nonces).astype("uint8")[nn_s:nn_e]
    print(N[0], N.shape, N.dtype)

    matcpa = MatCPA(T)
    
    # For k0 = 0
    k0 = 0
    h = []
    for i in range(nn_s, nn_e):
        n0 = (N[i][1] >> 7) & 1
        n1 = (N[i][9] >> 7) & 1
        v = fty0(k0, n0, n1)
        h.append(v)
    r0 = matcpa.do_cpa(nn_e-nn_s, np.array(h, dtype=np.uint8))

    # For k0 = 1
    k0 = 1
    h = []
    for i in range(nn_s, nn_e):
        n0 = (N[i][1] >> 7) & 1
        n1 = (N[i][9] >> 7) & 1
        v = fty0(k0, n0, n1)
        h.append(v)
    r1 = matcpa.do_cpa(nn_e-nn_s, np.array(h, dtype=np.uint8))

    # For HW(first byte of n0)
    h = []
    for i in range(nn_s, nn_e):
        h.append(HW(N[i][1]))
    ra = matcpa.do_cpa(nn_e-nn_s, np.array(h, dtype=np.uint8))

    # For HW(first byte of n0 ^ first byte of n1)
    h = []
    for i in range(nn_s, nn_e):
        h.append(HW(N[i][1] ^ N[i][9]))
    rb = matcpa.do_cpa(nn_e-nn_s, np.array(h, dtype=np.uint8))
    
    # For HW(first byte of y0)
    # h = []
    # for i in range(nn_s, nn_e):
    #     h.append(HW(y0(0x40, 0x01, 0x09, N[i][1], N[i][9])))
    # rc = matcpa.do_cpa(nn_e-nn_s, np.array(h, dtype=np.uint8))

    plt.rcParams.update({'font.size': 16})
    # plt.close()
    plt.figure(figsize=(8, 6))
    plt.xlabel("Sample")
    plt.ylabel("Absolute correlation")
    plt.plot(ra, color="lightgray")
    plt.plot(r0, color="blue")
    plt.show()
    # plt.close()
    plt.figure(figsize=(8, 6))
    plt.xlabel("Sample")
    plt.ylabel("Absolute correlation")
    plt.plot(rb, color="lightgray")
    plt.plot(r1, color="blue")
    plt.show()


