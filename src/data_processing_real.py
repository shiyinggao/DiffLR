import pandas as pd
import numpy as np
from scipy.stats import norm

def is_valid_table(df, expected_shape=(10, 11)):
    """ check full dataframe, first col AQ """
    return df.shape == expected_shape


def processingDataReal(data_directory, TableNumber,window_size, Stage_Size=40, Val_Size=4, Distribution='normal'):
    """ Process a single file and retain a 10x10 table by removing the first column """
    data_file = data_directory  + str(TableNumber) + '.csv'
    df = pd.read_csv(data_file )
    if is_valid_table(df):
        data = df.drop(df.columns[0], axis=1).to_numpy()
        
        N = data.shape[0]
        AQ = np.tile(np.array(range(N))[:, np.newaxis], (1, N))
        DQ = AQ.T

        LossMin = np.min(data[AQ + DQ < N]) * np.ones((N, N))
        LossMax = np.max(data[AQ + DQ < N]) * np.ones((N, N))
        LossNorm = (data - LossMin) / (LossMax - LossMin)

        LossMean = np.mean(data[AQ + DQ < N]) * np.ones((N, N))
        LossStdev = np.std(data[AQ + DQ < N]) * np.ones((N, N))

        top_rows = np.vstack([LossNorm[0, :], LossNorm[0, :]])
        LossNorm = np.vstack([top_rows, LossNorm])
        left_columns = np.hstack([LossNorm[:, [0]], LossNorm[:, [0]]])
        LossNorm = np.hstack([left_columns, LossNorm])

        submat_lossNorm = mysubmatrix(LossNorm, window_size=window_size)
        submat_lossNorm[:, -2:] = submat_lossNorm[:, -2:] - 2

        rows = submat_lossNorm.shape[0]  
        zeros_column = np.zeros((rows, 1))  

        AQ = AQ.flatten()
        DQ = DQ.flatten()

        trainValid_index = AQ + DQ < N
        test_index = ~trainValid_index

        trainValid_input = submat_lossNorm[trainValid_index, :]
        test_input = np.concatenate((submat_lossNorm[test_index, :-3], zeros_column[test_index], submat_lossNorm[test_index, -2:]), axis=1)
    
        output_Data = submat_lossNorm[:, -3]
        trainValid_output = output_Data[trainValid_index]
        test_output = output_Data[test_index]


        return LossMin.astype(np.float32),\
            LossMax.astype(np.float32),\
            trainValid_input.astype(np.float32),\
            trainValid_output.astype(np.float32),\
            test_input.astype(np.float32),\
            test_output.astype(np.float32)

    else:
        return 'Not Full'


def myflatten(x):
    x = np.flip(x, axis=1)
    N = x.shape[0]
    flatten_x = np.array([])
    for i in range(N-1, -N, -1):
        flatten_x = np.concatenate((flatten_x, np.flip(np.diagonal(x, i))))
    return flatten_x[:,np.newaxis]   # to Column Vector

def myunflatten(x):
    N = int(np.sqrt(x.shape[0]))
    mat = np.zeros((N,N))
    AQ = np.tile(np.array(range(N))[:, np.newaxis], (1, N))
    DQ = AQ.T
    AQ = myflatten(AQ).squeeze().astype(np.int64)
    DQ = myflatten(DQ).squeeze().astype(np.int64)
    mat[(AQ, DQ)] = x
    return mat

def mysubmatrix(lossMatrix, window_size = 3):
    N = lossMatrix.shape[0]
    submatrices = []
    for i in range(window_size-1, N):
        for j in range(window_size-1, N):
            submatrix = []
            for ii in range(window_size):
                for jj in range(window_size):
                    submatrix.append(lossMatrix[i - (window_size - 1) + ii,
                                                j - (window_size - 1) + jj])
            submatrix.append(i)
            submatrix.append(j)
            submatrix = np.array(submatrix)
            submatrices.append(submatrix)
    submatrices = np.array(submatrices)
    return submatrices

def preparingMapping(data, ccodp, components = 4, methods = 'Single Gaussian'):
    numCells = data.shape[0]
    if methods == 'Single Gaussian':
        alpha = np.zeros((numCells, components))
        LossMean = data[:,7][:,np.newaxis]
        LossStdev = data[:,8][:,np.newaxis]
        ccODP = ccodp[:,3][:,np.newaxis]
        dispersion = ccodp[:,4][:,np.newaxis]
        mu = (ccODP - LossMean)/LossStdev
        mu = np.tile(mu, (1, components))
        sigma = np.log(np.sqrt(ccODP * dispersion)/LossStdev)
        sigma = np.tile(sigma, (1, components))
        mappings = np.concatenate((alpha, mu, sigma), axis=1)
    return mappings




def mean_function(alpha, mu, sigma, nComponents = 4):
    nTrials = int(alpha.shape[1]/nComponents)
    return np.sum(alpha * mu, axis=1, keepdims = True)/nTrials

def logscore(data, alpha, mu, sigma, nComponents = 4):
    nTrials = int(alpha.shape[1] / nComponents)
    pdf_values = norm.pdf(np.tile(data, (1, alpha.shape[1])), loc=mu, scale=sigma)
    scores = np.sum(alpha * pdf_values, axis=1, keepdims=True) / nTrials
    return np.log(scores)

def sigma_mean(alpha, mu, sigma, nComponents = 4):
    nTrials = int(alpha.shape[1] / nComponents)
    means = mean_function(alpha, mu, sigma, nComponents)
    sigmaSquared = np.sum(alpha * (np.square(mu) + np.square(sigma)), axis=1, keepdims = True)/nTrials
    return np.sqrt(sigmaSquared - np.square(means))

def quantile_prediction(alpha, mu, sigma, nComponents = 4, quantile = 0.75, tolerance = 0.001, max_iter = 100):
    nTrials = int(alpha.shape[1] / nComponents)
    predMean = mean_function(alpha, mu, sigma, nComponents)
    predSigma = sigma_mean(alpha, mu, sigma, nComponents)
    quantile_est = predMean.copy()
    jump = predSigma.copy()
    predQuantile = np.zeros((alpha.shape[0], 1))

    tol = np.ones((alpha.shape[0], 1))
    incomplete = (np.abs(tol) > tolerance).squeeze()
    predQuantile[incomplete] = np.sum((alpha[incomplete]) * norm.cdf(np.tile(quantile_est[incomplete],(1, alpha.shape[1])), mu[incomplete], sigma[incomplete]), axis=1, keepdims = True) / nTrials

    old_predQuantile = predQuantile.copy() 
    tol = (predQuantile - quantile)
    jump[tol > 0] = -1 * jump[tol > 0]

    ticker = 0
    while ( (np.max(np.abs(tol)) > tolerance) and (ticker < max_iter)):
        incomplete = (np.abs(tol) > tolerance).squeeze()
        quantile_est[incomplete] = quantile_est[incomplete] + jump[incomplete]
        predQuantile[incomplete] = np.sum((alpha[incomplete]) * norm.cdf(np.tile(quantile_est[incomplete],(1, alpha.shape[1])), mu[incomplete], sigma[incomplete]), axis=1, keepdims = True) / nTrials
        tol[incomplete] = (predQuantile[incomplete] - quantile)
        went_above = np.logical_and((old_predQuantile < quantile), (predQuantile > quantile)).squeeze()
        went_below = np.logical_and((old_predQuantile > quantile), (predQuantile < quantile)).squeeze()
        jump[np.logical_and(went_above, incomplete)] = -0.5 * jump[np.logical_and(went_above, incomplete)]
        jump[np.logical_and(went_below, incomplete)] = -0.5 * jump[np.logical_and(went_below, incomplete)]
        ticker = ticker + 1
        old_predQuantile[incomplete] = predQuantile[incomplete].copy()

    return quantile_est

def quantile_loss(y_data, pred_quantile, quantile):
    quantile_loss = (y_data - pred_quantile) * (quantile - (y_data < pred_quantile))
    return quantile_loss

