import numpy as np 

def compute_moments(Sbar: np.darray, fisher: bool = False) -> dict:
    Sbar = np.asarray(Sbar, dtype=np.float64)
    n = Sbar.size
    if n < 2:
        raise ValueError("Need at least 2 samples to compute moments.")
    
    #Mean and central moments
    m1 = np.mean(Sbar)
    diffs = Sbar - m1
    v = np.mean(diffs ** 2)

    m3 = np.mean(diffs ** 3)
    m4 = np.mean(diffs ** 4)

    skew = m3 / (v ** 1.5)
    kurt = m4 / (v ** 2)
    if fisher:
        kurt -= 3.0

    return{"mean": m1, "var": v, "skew": skew, "kurt": kurt}
    