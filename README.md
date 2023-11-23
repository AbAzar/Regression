# Regression Implementation

This repository contains two Python scripts for implementing various regression techniques. The scripts are organized as follows:

## 1. norm_regression.py

This script implements regression using norm 0, 1, 2, and infinity norms. The following functions are provided:

- `predict(x, w, b)`: Predicts the output based on input features `x`, weight `w`, and bias `b`.
- `myplt(x, y, y_noise, y_predict, color)`: Generates a scatter plot of the data points and regression lines.
- `norm2(x, y_p)`: Implements norm 2 regression.
- `norm0(x, y_p)`: Implements norm 0 regression.
- `norm1(x, y_noise)`: Implements norm 1 regression.
- `norm_inf(x, y_noise)`: Implements norm infinity regression.
- `MSE(y, y_norm)`: Computes Mean Squared Error between original and predicted values.

The script concludes with an example of applying these regression techniques on synthetic data.

## 2. linear_regression.py

This script implements linear, ridge, and lasso regression. The functionalities include:

- `lin_reg(x, y)`: Linear regression function.
- `predict(x, w)`: Prediction function for linear regression.
- `ri_reg(lnd, x, y)`: Ridge regression function.
- `soft(a, dlt)`: Soft thresholding function for lasso regression.
- `lasso_reg(lnd, x, y)`: Lasso regression function.
- `sqt_err(ype, yte)`: Computes squared error between predicted and true values.
- `kernel_re_reg(x, y, lnd, metric='rbf', params={'gamma': 0.5})`: Kernel ridge regression.
- `predict_k(x_test, x_train, alpha, metric='rbf', params={"gamma": 0.5})`: Prediction function for kernel regression.

The script concludes with an example using the ailerons dataset and demonstrates the application of various regression techniques, including linear regression, ridge regression, lasso regression, and kernel ridge regression with different kernels.

## Instructions:

1. Ensure that the required libraries are installed using the following:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

2. Run the scripts using a Python interpreter.

Feel free to explore and modify the code to suit your specific regression needs.