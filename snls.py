import numpy as np
from scipy.optimize import minimize



class SNLS:
    
    def __init__(self):
        self.w_learned = None
        self.lu = (1, 4)

    def f_r(self, w, x):
        x1, x2 = x[:, 0], x[:, 1] 
        return (
            w[0] + w[1] * x1 + w[2] * x2 + w[3] * np.power(x1, w[4]) + 
            w[5] * np.power(x2, w[6]) + w[7] * np.exp(w[8] * x1 + w[9] * x2 + w[10])
        )

    def createData(self, w_true, num_points=100):
        x1 = np.random.uniform(self.lu[0], self.lu[1], num_points)
        x2 = np.random.uniform(self.lu[0], self.lu[1], num_points)
        X = np.column_stack([x1, x2])
        y_true = self.f_r(w_true, X)
        noise = np.random.normal(0.5, 2.0, num_points)
        y = y_true + noise
        X = np.column_stack((x1, x2))
        return X, y

    def loss(self, w, X, y):
        y_pred = self.f_r(w, X)
        return np.mean((y - y_pred) ** 2)

    def train(self, X, y):
        w_init = np.random.randn(11)
        result = minimize(self.x, w_init, args=(X, y), method='BFGS')
        self.w_learned = result.x
        return self.w_learned
    
    def test(self, X):
        y_pred = self.f_r(self.w_learned, X)
        return y_pred