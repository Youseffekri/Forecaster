
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Util.tools import dotdict
from forecasting.AR_YW import AR_YW
from forecasting.ARX import ARX
from forecasting.ARX_Symb import ARX_Symb
from forecasting.MHAttnRegressor import MHAttnRegressor

np.set_printoptions(linewidth=200)

def AR_YW_test(args, hh, y, method_type = 0):
    methods = ["sm_ols", "mle", "adjusted"]
    model = AR_YW(args, y, hh, methods[method_type])
    model.train()
    yf = model.forecast()
    qof = model.diagnose_all(yf)
    print(f"yf.shape = {yf.shape}")
    print(f"QoF:\n{qof}")
    print(f"params = [{', '.join(f'{x:.6f}' for x in model.get_params())}]")

    print("rollValidate")
    yf = model.rollValidate()
    qof = model.diagnose_all(yf, TnT = True)
    print(f"yf.shape = {yf.shape}")
    print(f"QoF:\n{qof}")


def ARX_test(args, hh, y, z = None, method_type=1):
    methods = ["sm_ols", "sk_lr"]
    model = ARX(args, y, hh, z, method=methods[method_type])
    # tForm = None #StandardScaler
    # model = ARX.rescale(args, y, hh, z, method=methods[method_type], tForm=tForm)
    np.savetxt("X_ARX.csv", model.X, delimiter=",", fmt="%.6f")

    model.train()
    yf = model.forecast()
    qof = model.diagnose_all(yf)
    print(f"X.shape = {model.X.shape}, yf.shape = {yf.shape}")
    print(f"QoF:\n{qof}")
    print(f"params = [{', '.join(f'{x:.6f}' for x in model.get_params())}]")

    print("rollValidate")
    yf = model.rollValidate()
    qof = model.diagnose_all(yf, TnT = True)
    print(f"yf.shape = {yf.shape}")
    print(f"QoF:\n{qof}")



def ARX_Symb_test(args, hh, y, z = None, cross=False, method_type=1):
    methods = ["sm_ols", "sk_lr"]
    ff = [lambda x: np.power(x, 1.5)]
    gg = []
    model = ARX_Symb(args, y, hh, z, cross, ff, gg, method=methods[method_type])
    # tForm = None # StandardScaler
    # model = ARX_Symb.rescale(args, y, hh, z, cross, method=methods[method_type], tForm=tForm)
    np.savetxt("X_ARX_Symb.csv", model.X, delimiter=",", fmt="%.6f")
    print(f"X.shape = {model.X.shape}")
    
    model.train()
    yf = model.forecast()
    qof = model.diagnose_all(yf)
    print(f"X.shape = {model.X.shape}, yf.shape = {yf.shape}")
    print(f"QoF:\n{qof}")
    # print(f"params = [{', '.join(f'{x:.6f}' for x in model.get_params())}]")
    
    print("rollValidate")
    yf = model.rollValidate()
    qof = model.diagnose_all(yf, TnT = True)
    print(f"yf.shape = {yf.shape}")
    print(f"QoF:\n{qof}")


def MHAttnRegressor_test(args, hh, y, z, cross):
    batch_size = 116
    cross = 1
    # model = ARX_Symb(args, y, hh, z, cross)
    tForm = None # StandardScaler
    model = ARX_Symb.rescale(args, y, hh, z, cross, tForm=tForm)

    X, y = model.X, model.y.reshape(-1, 1)
    print(f"X.shape = {X.shape}")
    X_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.float32)
    train_ratio = 1.0
    num_train = int(len(X_torch) * train_ratio)   
    X_train, X_test = X_torch[:num_train], X_torch[num_train:]
    y_train, y_test = y_torch[:num_train], y_torch[num_train:]

    train_dataset = TensorDataset(X_train, y_train)
    # test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    heads = 4
    d_model = 72
    model = MHAttnRegressor(input_dim=X_train.shape[1], d_model=d_model, num_heads=heads)
    criterion = nn.MSELoss()
    # criterion = nn.HuberLoss(delta=2.0)

    model.trainX(train_loader, criterion)
    heatMap = model.testX(train_loader, criterion)
    # heatMap = model.testX(test_loader, criterion)

    num_fs = 20
    feature_scores = heatMap.mean(dim=0).numpy()
    feature_scores_withIntercept = np.insert(feature_scores, 0, 1.0)
    feature_scores_sorted = sorted(list(enumerate(feature_scores_withIntercept)), key=lambda x: x[1], reverse=True)
    selected = [(i, round(float(score), 6)) for i, score in feature_scores_sorted[:num_fs]]
    selected_idx = [i for i, _ in selected]
    print(f"selected for Scala: {selected}")
    print(selected_idx)

if __name__ == "__main__":

    dataLoad = pd.read_csv('data/covid_19_weekly.csv')
    data_y = dataLoad[['new_deaths']].iloc[:116].reset_index(drop=True)
    data_z = dataLoad[['icu_patients', 'hosp_patients']].iloc[:116].reset_index(drop=True)
    y  = data_y['new_deaths'].to_numpy()
    # z = data_z[['icu_patients', 'hosp_patients']].to_numpy()
    z = data_z[['icu_patients']].to_numpy()

    args = dotdict()
    args.skip = 2
    args.spec = 1
    args.p = 2
    args.q = 1
    hh = 6
    cross = False
    # AR_YW_test(args, hh, y, method_type=1)
    # AR_YW_test(args, hh, y, method_type=2)
    AR_YW_test(args, hh, y, method_type=0)
    # ARX_test(args, hh, y, method_type=0)
    # ARX_test(args, hh, y, z)
    # ARX_Symb_test(args, hh, y, z, cross)
    # MHAttnRegressor_test(args, hh, y, z, cross)


    







    
    
