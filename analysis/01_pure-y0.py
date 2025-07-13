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
    
    R = []
    for k in range(4):
        k0 = (k >> 1) & 1
        k1 = k & 1
        h = []
        for i in range(nn_s, nn_e):
            n0 = (N[i][0] >> 7) & 1
            n1 = (N[i][8] >> 7) & 1
            v = y0(1, k0, k1, n0, n1)
            h.append(v)
        r = matcpa.do_cpa(nn_e-nn_s, np.array(h, dtype=np.uint8))
        R.append(r)

    (yb, yt) = (-0.01,0.22)
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(8, 6))
    plt.xlabel("Sample")
    plt.ylabel("Absolute correlation")
    plt.ylim((yb, yt))
    plt.plot(R[0], color="blue")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.xlabel("Sample")
    plt.ylabel("Absolute correlation")
    plt.ylim((yb, yt))
    plt.plot(R[1], color="blue")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.xlabel("Sample")
    plt.ylabel("Absolute correlation")
    plt.ylim((yb, yt))
    plt.plot(R[2], color="red")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.xlabel("Sample")
    plt.ylabel("Absolute correlation")
    plt.ylim((yb, yt))
    plt.plot(R[3], color="red")
    plt.show()


