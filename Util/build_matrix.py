

import numpy as np

q_mean = 2          # number of prior values for mean

def backcast(y_: np.ndarray, i: int = 0):
    q_m = q_mean
    ww = np.arange(1, q_m+1)
    b = ww/ww.sum()
    yy = y_[i:q_m+i][::-1]
    return np.dot(b, yy)

def backfill(zj: np.ndarray):
    z_j = zj.copy().astype(float)
    z_j = np.concatenate(([0.0], z_j))
    ii = np.argmax(z_j != 0.0) - 1

    for i_ in range(ii+1):
        i = ii - i_
        z_j[i] = backcast(z_j, i)
    return z_j[1:]

def backfill_matrix(z: np.ndarray):
    z_bfill = z.copy().astype(float)
    for j in range(z.shape[1]): z_bfill[:, j] = backfill(z[:, j])
    return z_bfill

def build_trend_matrix(m, spec = 1, lwave = 20):
    m2 = m / 2.0
    w = (2 * np.pi) / lwave  
    x = np.zeros((m, spec - 1))  
    t_0m = np.arange(m)

    if spec >= 2:
        x[:, 0] = t_0m / m                        
    if spec >= 3:
        x[:, 1] = ((t_0m - m2) ** 2) / (m2 ** 2)  
    if spec >= 4:
        x[:, 2] = np.sin(t_0m * w)                
    if spec == 5:
        x[:, 3] = np.cos(t_0m * w)                
    return x

def build_lagged_matrix(x, lag):
    first = x[0]
    ones_ = first * np.ones(lag)
    xRow = np.concatenate((ones_, x[:-1]))
    xx = np.column_stack([xRow[i:len(xRow) - lag + i + 1] for i in range(lag)])
    return xx

def build_matrix_4ts(y, spec = 1, p=2, z = None, q = 0, tForms={"tForm_y": None}):
    if tForms["tForm_y"] is not None:
        y = tForms["tForm_y"].fit_transform(y.reshape(-1, 1)).flatten()
    X = build_lagged_matrix(y, p)

    if z is not None:
        z = backfill_matrix(z)
        if tForms["tForm_y"] is not None:
            z = tForms["tForm_exo"].fit_transform(z)
        x_exo = np.column_stack([build_lagged_matrix(z[:, j], q) for j in range(z.shape[1])])
        X = np.column_stack((X, x_exo))

    if spec > 1:
        xt = build_trend_matrix(len(y), spec)
        X = np.column_stack((xt, X))
    return X

def build_matrix_symbolic_4ts(y, spec = 1, p=2, z = None, q = 0, cross: bool = False, tForms={"tForm_y": None}):
    
    y_fEndo = np.column_stack([f(y) for f in tForms["fEndo"]])

    if tForms["tForm_y"] is not None:
        y       = tForms["tForm_y"].fit_transform(y.reshape(-1, 1)).flatten()
        y_fEndo = tForms["tForm_endo"].fit_transform(y_fEndo)
    y_fEndo = np.column_stack((y, y_fEndo))
    X = np.column_stack([build_lagged_matrix(y_fEndo[:, j], p) for j in range(y_fEndo.shape[1])])

    if z is not None:
        z_bfill = backfill_matrix(z)
        if len(tForms["fExo"]) > 0:
            x_exo = np.column_stack([f(z_bfill) for f in tForms["fExo"]])
            x_exo = np.column_stack((z_bfill, x_exo))
        else:
            x_exo = z_bfill.copy()

        if cross:
            yz = np.column_stack([y*z_bfill[:, j] for j in range(z.shape[1])])
            x_exo  = np.column_stack((x_exo, yz))

        if tForms["tForm_y"] is not None:
            x_exo = tForms["tForm_exo"].fit_transform(x_exo)
        x_exo = np.column_stack([build_lagged_matrix(x_exo[:, j], q) for j in range(x_exo.shape[1])])
        X = np.column_stack((X, x_exo))

    if spec > 1:
       xt = build_trend_matrix(len(y), spec)
       X = np.column_stack((xt, X))
    return X

def buildYf(y, hh):
    y_fcast = np.zeros((y.shape[0], hh))
    time = np.arange(0, y.shape[0]).reshape(-1, 1)
    yf = np.hstack([y.reshape(-1, 1), y_fcast, time])
    return yf