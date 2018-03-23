import scipy.io as sio
import os
import numpy as np
import pywt

matfile_X = 'hyperspectral_datas/salina/data/Salinas_corrected.mat'
matfile_y = 'hyperspectral_datas/salina/data/Salinas_gt.mat'
dataX = sio.loadmat(matfile_X)['salinas_corrected']
datay = sio.loadmat(matfile_y)['salinas_gt']

salinaNpyfileSpectral = 'salina_spectral.npy'
salinaNpyfileNeighborSpatial = 'salina_neighbor_spatial.npy'
salinaNpyfileWaveletSpatial = 'salina_wavelet_spatial.npy'
salinaNpyfileY = 'salina_y.npy'

def Z_ScoreNormalization(x):
    mu = np.average(x)
    sigma = np.std(x)
    x = (x - mu) / sigma
    return x
    
# 处理光谱特征
def getSalinaSpectral():
    spectral = []
    y = []
    for i in range(512):
        for j in range(217):
            if datay[i][j] == 0:
                continue
            vec = Z_ScoreNormalization(np.array(dataX[i, j, :]))
            spectral.append(vec)
            y.append(datay[i][j]-1)
    spectral = np.array(spectral).T
    y = np.array(y)
    print(spectral.shape)
    print(y.shape)
    np.save(salinaNpyfileSpectral, spectral)
    np.save(salinaNpyfileY, y)

    return spectral, y

def getSalinaNeighbor(n):
    spatial = []
    label = []
    for x in range(512):
        print(x)
        for y in range(217):
            if datay[x][y] > 0:
                vec = []
                W = []
                for i in range(-int(n / 2), int(n / 2) + 1):
                    for j in range(-int(n / 2), int(n / 2) + 1):
                        if i == 0 and j == 0:
                            continue
                        nx = x + i
                        ny = y + j
                        if nx < 0 or nx >= 512 or ny < 0 or ny >= 217:
                            continue
                        vec.append(dataX[nx][ny][:])
                num = len(vec)
                dist = [0]*num
                for i in range(num):
                    dist[i] = np.linalg.norm(np.array(vec[i]) - np.array(dataX[x, y, :]))
                avg_dis = sum(dist)/num
                for i in range(num):
                    W.append(np.exp(-dist[i] / avg_dis))
                label.append(datay[x][y]-1)
                fea = np.zeros(204)
                for i in range(num):
                    fea += W[i]*vec[i]
                fea /= sum(W)
                fea = Z_ScoreNormalization(fea)
                spatial.append(fea)
    spatial = np.array(spatial).T
    label = np.array(label)
    print('spatial-shape:', spatial.shape)
    print('y-shape:', label.shape)
    np.save(salinaNpyfileNeighborSpatial, spatial)

    return spatial

def wavelet(data):
    cA, (cH, cV, cD) = pywt.dwt2(data, 'haar')

    return cA.reshape(-1)

def getSalinaWavelet(n):
    spatial = []
    y = []
    for i in range(512):
        print(i)
        for j in range(217):
            if datay[i][j] != 0:
                fea = []
                for band in range(204):
                    data = []
                    for nx in range(i-int(n/2)+1, i+int(n/2)+1):
                        for ny in range(j-int(n/2)+1, j+int(n/2)+1):
                            xx = nx
                            yy = ny
                            if xx < 0:
                                xx = 0
                            if xx >= 512:
                                xx = 511
                            if yy < 0:
                                yy = 0
                            if yy >= 217:
                                yy = 216
                            data.append(dataX[xx][yy][band])
                    vec = wavelet(np.array(data).reshape((n,n)))
                    fea.append(vec)
                y.append(datay[i][j]-1)
                fea = np.array(fea).reshape(-1)
                fea = Z_ScoreNormalization(fea)
                spatial.append(fea)
    spatial = np.array(spatial).T
    y = np.array(y)
    print('spatial-shape:', spatial.shape)
    print('y-shape:', y.shape)
    np.save(salinaNpyfileWaveletSpatial, spatial)

def getSalinaDataset(spatialFeature):
    if os.path.exists(salinaNpyfileSpectral):
        X_spectral = np.load(salinaNpyfileSpectral)
        Y = np.load(salinaNpyfileY)
    else:
        X_spectral, Y = getSalinaSpectral()

    if spatialFeature == 'neighbor':
        if os.path.exists(salinaNpyfileNeighborSpatial):
            X_spatial = np.load(salinaNpyfileNeighborSpatial)
        else:
            X_spatial = getSalinaNeighbor(9)
    elif spatialFeature == 'wavelet':
        if os.path.exists(salinaNpyfileWaveletSpatial):
            X_spatial = np.load(salinaNpyfileWaveletSpatial)
        else:
            X_spatial = getSalinaWavelet(4)
    else:
        return X_spectral, Y
        # exit('spatial feature error!')

    return X_spectral, X_spatial, Y