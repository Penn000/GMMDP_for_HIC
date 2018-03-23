import scipy.io as sio
import numpy as np
import os
import pywt

def Z_ScoreNormalization(x):
    mu = np.average(x)
    sigma = np.std(x)
    x = (x - mu) / sigma
    return x

indianMatfileY = 'hyperspectral_datas/indian_pines/data/Indian_pines_gt.mat'
indianMatfileX = 'hyperspectral_datas/indian_pines/data/Indian_pines_corrected.mat'
indianDataX = sio.loadmat(indianMatfileX)['indian_pines_corrected']
indianDataY = sio.loadmat(indianMatfileY)['indian_pines_gt']

indianNpyfileSpectral = 'indian_pines_spectral.npy'
indianNpyfileNeighborSpatial = 'indian_pines_neighbor_spatial.npy'
indianNpyfileWaveletSpatial = 'indian_pines_wavelet_spatial.npy'
indianNpyfileY = 'indian_pines_y.npy'

relabel = {'2': 0, '3': 1, '5': 2, '6': 3, '8': 4, '10': 5, '11': 6, '12': 7, '14': 8}
selected_class = [2, 3, 5, 6, 8, 10, 11, 12, 14]

# 处理光谱特征
def getIndianSpectral():
    spectral = []
    y = []
    for i in range(145):
        for j in range(145):
            if indianDataY[i][j] in selected_class:
                vec = Z_ScoreNormalization(np.array(indianDataX[i, j, :]))
                # vec = np.array(indianDataX[i, j, :])
                spectral.append(vec)
                y.append(relabel[str(indianDataY[i][j])])
    spectral = np.array(spectral).T # 一列一样本
    y = np.array(y)
    print(spectral.shape)
    print(y.shape)
    np.save(indianNpyfileSpectral, spectral)
    np.save(indianNpyfileY, y)

    return spectral, y

def getIndianNeighbor(n):
    spatial = []
    label = []
    for x in range(145):
        print(x)
        for y in range(145):
            if indianDataY[x][y] in selected_class:
                vec = []
                W = []
                for i in range(-int(n / 2), int(n / 2) + 1):
                    for j in range(-int(n / 2), int(n / 2) + 1):
                        if i == 0 and j == 0:
                            continue
                        nx = x + i
                        ny = y + j
                        if nx < 0 or nx >= 145 or ny < 0 or ny >= 145:
                            continue
                        vec.append(indianDataX[nx][ny][:])
                num = len(vec)
                dist = [0]*num
                for i in range(num):
                    dist[i] = np.linalg.norm(np.array(vec[i]) - np.array(indianDataX[x, y, :]))
                avg_dis = sum(dist)/num
                for i in range(num):
                    W.append(np.exp(-dist[i] / avg_dis))
                label.append(relabel[str(indianDataY[x][y])])
                fea = np.zeros(200)
                for i in range(num):
                    fea += W[i]*vec[i]
                fea /= sum(W)
                fea = Z_ScoreNormalization(fea)
                spatial.append(fea)
    spatial = np.array(spatial).T
    label = np.array(label)
    print('spatial-shape:', spatial.shape)
    print('y-shape:', label.shape)
    np.save(indianNpyfileNeighborSpatial, spatial)

def wavelet(data):
    cA, (cH, cV, cD) = pywt.dwt2(data, 'haar')

    return cA.reshape(-1)

def getIndianWavelet(n):
    spatial = []
    y = []
    for i in range(145):
        print(i)
        for j in range(145):
            if indianNpyfileY[i][j] in selected_class:
                fea = []
                for band in range(200):
                    data = []
                    for nx in range(i-int(n/2)+1, i+int(n/2)+1):
                        for ny in range(j-int(n/2)+1, j+int(n/2)+1):
                            xx = nx
                            yy = ny
                            if xx < 0:
                                xx = 0
                            if xx >= 145:
                                xx = 144
                            if yy < 0:
                                yy = 0
                            if yy >= 145:
                                yy = 144
                            data.append(indianDataX[xx][yy][band])
                    vec = wavelet(np.array(data).reshape((n,n)))
                    fea.append(vec)
                y.append(relabel[str(indianNpyfileY[i][j])])
                fea = np.array(fea).reshape(-1)
                fea = Z_ScoreNormalization(fea)
                spatial.append(fea)
    spatial = np.array(spatial).T
    y = np.array(y)
    print('spatial-shape:', spatial.shape)
    print('y-shape:', y.shape)
    np.save(indianNpyfileWaveletSpatial, spatial)

    return spatial

def getIndianDataset(spatialFeature):
    if os.path.exists(indianNpyfileSpectral):
        X_spectral = np.load(indianNpyfileSpectral)
        Y = np.load(indianNpyfileY)
    else:
        X_spectral, Y = getIndianSpectral()

    if spatialFeature == 'neighbor':
        if os.path.exists(indianNpyfileNeighborSpatial):
            X_spatial = np.load(indianNpyfileNeighborSpatial)
        else:
            X_spatial = getIndianNeighbor(9)
    elif spatialFeature == 'wavelet':
        if os.path.exists(indianNpyfileNeighborSpatial):
            X_spatial = np.load(indianNpyfileNeighborSpatial)
        else:
            X_spatial = getIndianWavelet(4)
    else:
        exit('spatial feature error!')

    return X_spectral, X_spatial, Y