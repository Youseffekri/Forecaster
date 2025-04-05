
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
from Util.data_loader import Data_Loader
from modeling.forecasting.ar_yw import AR_YW
from modeling.forecasting.arx import ARX
from modeling.forecasting.arx_d import ARX_D
from modeling.forecasting.arx_symb import ARX_Symb
from modeling.forecasting.arx_symb_d import ARX_Symb_D
from modeling.neuralnet.neuralnet_3l import NeuralNetwork_3L
from modeling.forecasting.neuralforecasting.mhattn_regressor import MHAttn_Regressor

np.set_printoptions(linewidth=200, precision=4, suppress=True)

def AR_YW_test(args, hh, y, method="sm_ols"):
    model = AR_YW(args, y, hh, method)
    model.inSample_Test(showParams=False, showYf=False)
    model.trainNtest_Test(showParams=False, showYf=False)


def ARX_test(args, hh, y, xe = None, method="sm_ols"):
    # model = ARX(args, y, hh, xe, method=method)
    model = ARX.rescale(args, y, hh, xe, method=method, tForm=StandardScaler)
    model.inSample_Test(showParams=True, showYf=True)
    model.trainNtest_Test(showParams=True, showYf=True)


def ARX_D_test(args, hh, y, xe = None, method="sm_ols"):
    # model = ARX_D(args, y, hh, xe, method=method)
    model = ARX_D.rescale(args, y, hh, xe, method=method, tForm=StandardScaler)
    model.inSample_Test(showParams=False, showYf=False)
    model.trainNtest_Test(showParams=False, showYf=False)


def ARX_Symb_test(args, hh, y, xe = None, method="sm_ols"):
    ff = [lambda x: np.power(x, 1.5)]
    gg = []
    # model = ARX_Symb(args, y, hh, xe, ff, gg, method=method)
    model = ARX_Symb.rescale(args, y, hh, xe, ff, gg, method=method, tForm=StandardScaler)
    model.inSample_Test(showParams=False, showYf=False)
    model.trainNtest_Test(showParams=False, showYf=False)


def ARX_Symb_D_test(args, hh, y, xe = None, method="sm_ols"):
    ff = [lambda x: np.power(x, 1.5)]
    gg = []
    # model = ARX_Symb_D(args, y, hh, xe, ff, gg, method=method)
    model = ARX_Symb_D.rescale(args, y, hh, xe, ff, gg, method=method, tForm=StandardScaler)
    model.inSample_Test(showParams=False, showYf=False)
    model.trainNtest_Test(showParams=False, showYf=False)


def NN_3L_test(args, hh, y, xe = None, method="sm_ols"):
    model = ARX_D.rescale(args, y, hh, xe, method=method, tForm=StandardScaler)
    data = Data_Loader(model.X, model.Y, TnT=True)
    X_train, X_test, Y_train, Y_test = data["train_input"], data["test_input"], data["train_label"], data["test_label"]
    
    nn_model = NeuralNetwork_3L(input_dim=X_train.shape[1], output_dim=Y_train.shape[1])
    nn_model.train(X_train, Y_train)
    Yp, _ = nn_model.test(X_test, Y_test)

    from Util.tools import diagnose
    for j in range(hh):
        y_true = model._yForm.inverse_transform(Y_test[:, j:j+1].numpy()).flatten()
        y_pred = model._yForm.inverse_transform(Yp[:, j:j+1].numpy()).flatten()
        print(f"h = {j+1}, qof =")
        print(diagnose(y_true, y_pred))


def MHAttnRegressor_test(args, hh, y, xe):
    batch_size = 116
    # model = ARX_Symb(args, y, hh, xe)
    model = ARX_Symb.rescale(args, y, hh, xe, tForm=StandardScaler)

    X, y = model.X, model.y.reshape(-1, 1)
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
    model = MHAttn_Regressor(input_dim=X_train.shape[1], d_model=d_model, num_heads=heads)
    criterion = nn.MSELoss()
    # criterion = nn.HuberLoss(delta=2.0)

    model.trainX(train_loader, criterion)
    heatMap = model.testX(train_loader, criterion)
    # heatMap = model.testX(test_loader, criterion)

    num_fs = 20
    feature_scores = heatMap.mean(dim=0).numpy()
    # feature_scores_withIntercept = np.insert(feature_scores, 0, 1.0)
    feature_scores_sorted = sorted(list(enumerate(feature_scores)), key=lambda x: x[1], reverse=True)
    selected = [(i, round(float(score), 6)) for i, score in feature_scores_sorted[:num_fs]]
    selected_idx = [i for i, _ in selected]
    print(f"selected columns: {selected}")
    print(selected_idx)

if __name__ == "__main__":

    dataLoad = pd.read_csv('data/covid_19_weekly.csv')
    data = dataLoad[['new_deaths', 'icu_patients', 'hosp_patients']].iloc[:116].reset_index(drop=True)
    y  = data['new_deaths'].to_numpy()
    # xe = data[['icu_patients', 'hosp_patients']].to_numpy()
    xe = data[['icu_patients']].to_numpy()

    args = dotdict()
    args.skip = 2
    args.spec = 1
    args.p = 6
    args.q = 4
    args.cross = False
    hh = 6

    methods = ["sm_ols", "mle", "adjusted"]
    AR_YW_test(args, hh, y)
    AR_YW_test(args, hh, y, method=methods[1])
    AR_YW_test(args, hh, y, method=methods[2])

    ARX_test(args, hh, y, xe)
    ARX_test(args, hh, y, xe, method="sk_lr")
    ARX_D_test(args, hh, y, xe)
    ARX_D_test(args, hh, y, xe, method="sk_lr")

    ARX_Symb_test(args, hh, y, xe)
    ARX_Symb_D_test(args, hh, y, xe, method="sk_lr")

    NN_3L_test(args, hh, y, xe)

    MHAttnRegressor_test(args, hh, y, xe)

    







    
    
