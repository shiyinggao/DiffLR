import os
import torch
import random 
import numpy as np
from torch.utils.data import Dataset

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class MyDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        x = self.input_data[idx]
        y = self.target_data[idx]
        return x, y

def str2bool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))

# performance metrics
def mae(
        x,
        x_hat
    ):
    r"""
    Compute the Mean absolute Error of the data along the given axis.

    Params:
    x : array_like, The ground truths
    x_hat : array_like, The predict values 
    """
    x = np.asarray(x)
    x_hat = np.asarray(x_hat)
    if x.shape != x_hat.shape:
        x_hat.squeeze()
    diff = (x - x_hat)
    mae = np.mean(np.abs(diff))

    return mae 

def rmse(x, x_hat):
    """
    Calculate the Root Mean Square Error (RMSE)
    
    Params:
    x: array_like, The true values.
    x_hat: array_like, The predicted values.
    """
    x = np.asarray(x)
    x_hat = np.asarray(x_hat)
    if x.shape != x_hat.shape:
        x_hat.squeeze()
    squared_diff = (x- x_hat) ** 2
    mse = np.mean(squared_diff)
    rmse = np.sqrt(mse)
    
    return rmse

def mape(x, x_hat):
    """
    Calculate the Mean Absolute Percentage Error (MAPE)
    Params:
    x: array_like, The true values.
    x_hat: array_like, The predicted values.
    """
    x = np.array(x)
    x_hat = np.array(x_hat)
    if x.shape != x_hat.shape:
        x_hat.squeeze()
    absolute_percentage_error = np.abs(x - x_hat) / (np.abs(x) + 1e-3)
    
    mape_value = np.mean(absolute_percentage_error) * 100.0
    return mape_value 

def evaluate(x, x_hat):
    """Evaluate prediction result by MAE RMSE MAPE
    Params:
    x: array_like, The true values.
    x_hat: array_like, The predicted values.
    """
    mae_result = mae(x, x_hat)
    rmse_result = rmse(x, x_hat)
    mape_result = mape(x, x_hat)

    return mae_result, rmse_result, mape_result


def ecrps(x, x_hat):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) using empirical CDF.
    
    Parameters:
    x : array_like
        The true values (shape: [n_obs]).
    x_hat : array_like
        The predicted values (multiple samples per observation, shape: [n_obs, n_samples]).
    
    Returns:
    crps_score : float
        The average CRPS score across all observations.
    """
    crps_values = []
    
    for i in range(len(x)):
        sorted_x_hat = np.sort(x_hat[i])
        
        cdf_pred = np.arange(1, len(sorted_x_hat) + 1) / len(sorted_x_hat)
        
        crps_value = 0
        for j, pred in enumerate(sorted_x_hat):
            indicator = 1 if pred >= x[i] else 0
            crps_value += (cdf_pred[j] - indicator) ** 2
        
        crps_value /= len(sorted_x_hat)
        crps_values.append(crps_value)
    
    return np.mean(crps_values)