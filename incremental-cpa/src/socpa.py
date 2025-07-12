from pearson import pearson_v5
from utils import load_trace_matrix
from model import load_guess_matrix_z, load_guess_matrix_y
from ranking import ranking_cpa
from tqdm import tqdm
import numpy as np
import pickle
import os

def socpa_v9(config, trace_indexes):
    nk = config.nk
    nt = config.nt
    nn = config.nn
    nw = config.nw
    ns = config.ns
    W = int(nw*(nt - (nw-1)/2))
    _dtype = config.data_type

    if nn % ns != 0: raise ValueError(f"Number of traces must be a multiple of step")

    # Files of checkpoints
    checkpoints = [int(v.split(".")[0]) for v in sorted(os.listdir(config.path_to_checkpoints))
                   if v != ".DS_Store"]
    
    # Determine start number of traces
    if len(checkpoints) == 0: 
        cpdata = {}
        nn_cp = 0
        s01 = np.zeros(nt, dtype=_dtype)
        # s02 = np.zeros(nt, dtype=np.float32) # Reuse s01
        s03 = np.zeros(nk, dtype=_dtype)
        s04 = np.zeros(W, dtype=_dtype)
        s05 = np.zeros((nt,nk), dtype=_dtype)
        s06 = np.zeros(nt, dtype=_dtype)
        # s07 = np.zeros((nt,nk), dtype=np.float32) Reuse s05
        # s08 = np.zeros(nt, dtype=np.float32) # Reuse s06
        s09 = np.zeros(nk, dtype=_dtype)
        s10 = np.zeros((nk,W), dtype=_dtype)
        s11 = np.zeros(W, dtype=_dtype)
        s12 = np.zeros(W, dtype=_dtype)
        s13 = np.zeros(W, dtype=_dtype)
    else:
        nn_cp = checkpoints[-1]
        with open(f"{config.path_to_checkpoints}/{nn_cp:07d}.pkl", 'rb') as f: 
            cpdata = pickle.load(f)
        if nn <= nn_cp:
            print(f"{nn} Already calculated and stored in file of checkpoints")
            print(f"[INFO] Incremental cpa with {nn_cp} traces:")
            ranked_candidates = ranking_cpa(cpdata[nn_cp]["rho"], nr=config.nr)
            print(f"[INFO] Ranking key candidate:")
            for i, item in enumerate(ranked_candidates): 
                print(f"Rank {i:2d}: {item[1]:4d} - {item[0]:.06f}")
            return
        
        s01 = cpdata["s01"]
        # s02 = cpdata["s02"]
        s03 = cpdata["s03"]
        s04 = cpdata["s04"]
        s05 = cpdata["s05"]
        s06 = cpdata["s06"]
        # s07 = cpdata["s07"]
        # s08 = cpdata["s08"]
        s09 = cpdata["s09"]
        s10 = cpdata["s10"]
        s11 = cpdata["s11"]
        s12 = cpdata["s12"]
        s13 = cpdata["s13"]


    # Compute cpa with step
    for i in range((nn-nn_cp)//ns):
        nn_cur = nn_cp+(i+1)*ns
        rho = np.zeros((nk,W), dtype=_dtype)
        print(f"[INFO] Incremental cpa with {nn_cur} traces:")

        # Randomize indexes of traces
        step_indexes = trace_indexes[i*ns:(i+1)*ns]

        # Load HW matrix for all guesses
        if   config.selection_function in ["y0", "y1", "y4"]: K = load_guess_matrix_y(config, step_indexes)
        elif config.selection_function in ["z0", "z1", "z4"]: K = load_guess_matrix_z(config, step_indexes)
        else: raise ValueError("Invalid selection function")

        # Load matrix of traces
        T = load_trace_matrix(config, step_indexes)

        # Run incremental CPA
        for i in tqdm(range(nk)):
            s03[i] += np.sum(K[i])
            s09[i] += np.sum(K[i] ** 2)

        for i in tqdm(range(nt)):
            s01[i] += np.sum(T[i])
            s06[i] += np.sum(T[i] ** 2)
            for j in range(nk):
                s05[i,j] += np.dot(T[i], K[j])

        m = 0
        for i in tqdm(range(nt)):
            for j in range(i, min(nt,i+nw)):
                t04 = T[i] * T[j]
                s04[m] += np.sum(t04)
                s11[m] += np.dot(T[i] ** 2, T[j] ** 2)
                s12[m] += np.dot(T[i] ** 2, T[j])
                s13[m] += np.dot(T[i], T[j] ** 2)

                for l in range(nk):
                    s10[l,m] += np.dot(t04, K[l])
                    rho[l,m] = pearson_v5(nn_cur, s01[i], s01[j], s03[l], s04[m], s05[i,l], s06[i], s05[j,l], s06[j], s09[l], s10[l,m], s11[m], s12[m], s13[m])
                        
                m += 1

        # Show ranking for key candidates
        ranked_candidates = ranking_cpa(rho, nr=config.nr)
        print(f"[INFO] Ranking key candidate:")
        for i, item in enumerate(ranked_candidates): 
            print(f"Rank {i:2d}: {item[1]:4d} - {item[0]:.06f}")

        # Saving results to file
        cpdata = {"s01": s01, "s03": s03, "s04": s04, "s05": s05, "s06": s06, "s09": s09, "s10": s10, "s11": s11, "s12": s12, "s13": s13, "rho": rho}
        with open(f"{config.path_to_checkpoints}/{nn_cur:07d}.pkl", 'wb') as f:  
            pickle.dump(cpdata, f)
