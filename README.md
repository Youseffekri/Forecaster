# Forecaster

A modular time series forecasting framework that integrates symbolic regression, ARX models, and attention-based feature selection. Designed for multi-step forecasting and model evaluation.

## Features

- AR and ARX models with customizable lag structures
- Symbolic regression with nonlinear transformations
- Attention-based feature selection using PyTorch
- Rolling window validation and quality of forecast metrics

## Getting Started

1. Git clone the repository
```bash
git clone https://github.com/Youseffekri/Forecaster.git
```


2. Install dependencies:

```bash
cd Forecaster
pip install -r requirements.txt
```

3. Run the example script:

```bash
python .\forecasting\Example_Covid19.py
```

## Models

- `AR_YW`: Autoregressive model using Yule-Walker estimation
- `ARX`, `ARX_D`: AR model with exogenous inputs
- `ARX_Symb`, `ARX_Symb_D`: Symbolic regression-based ARX
- `MHAttnRegressor`: Multi-head attention for feature selection

## Evaluation Metrics

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (R² and Adjusted R²)
- Symmetric Mean Absolute Percentage Error (SMAPE)

## License

MIT License

## Author

Yousef Fekri Dabanloo  
March 2025