import numpy as np
import IndianDatasetProcess
import SalinaDatasetProcess
from sklearn.preprocessing import StandardScaler

def PCA_reduction(X_multiview, n_components):
    n_views = len(X_multiview)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    for vi in range(n_views):
        X = X_multiview[vi].T
        X_multiview[vi] = pca.fit_transform(X).T
    return X_multiview

def split_dataset_multi(X_multiview, label_multiview, size):
    from sklearn.model_selection import train_test_split
    n_views = len(X_multiview)
    X_trainset = [0] * n_views
    X_testset = [0] * n_views
    y_trainset = [0] * n_views
    y_testset = [0] * n_views
    for vi in range(n_views):
        X = X_multiview[vi].T
        y = list(label_multiview[vi])
        # print('X_multiview[vi]:', X.T.shape)
        # print('label_multiview[vi]:', len(y))
        n_class = len(np.unique(label_multiview[vi]))  # 种类个数
        n_samples = len(label_multiview[vi])
        # print('*',n_samples)
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        for ci in range(n_class):
            id = [j for j in range(n_samples) if ci == y[j]]
            XX = []
            yy = []
            for j in id:
                XX.append(X[j])
                yy.append(y[j])
            x_X_train, x_X_test, y_y_train, y_y_test = train_test_split(XX, yy, test_size=size, random_state=0)
            X_train.append(x_X_train)
            X_test.append(x_X_test)
            y_train.append(y_y_train)
            y_test.append(y_y_test)

        X_trainset[vi] = np.row_stack(X_train).T
        X_testset[vi] = np.row_stack(X_test).T
        y_trainset[vi] = np.hstack(y_train)
        y_testset[vi] = np.hstack(y_test)

    return X_trainset, X_testset, y_trainset, y_testset

def projecting(X_multiview, W):
    n_views = len(X_multiview)
    X_transform = [0] * n_views
    for vi in range(n_views):
        n_sample = X_multiview[vi].shape[1]
        tran_view = [0] * n_sample
        for i in range(n_sample):
            sample = [x for x in (W[vi].T.dot(X_multiview[vi][:, i])).T]
            tran_view[i] = sample
            # print('tran_sample', tran_sample[i].shape)
        # print('tran_view: ',tran_view)
        X_transform[vi] = np.array(tran_view).T.tolist()

    return X_transform

def classifier_by_svm(X_trainset, y_trainset, X_testset, y_testset, n_views, gamma):
    X = np.hstack((X_trainset[i]) for i in range(n_views)).T
    label = np.hstack((y_trainset[i]) for i in range(n_views)).T

    from sklearn.svm import SVC
    svc = SVC(kernel='rbf', gamma=gamma)
    svc.fit(X, label)

    X = np.array(X_testset[1]).T
    test_label = np.array(y_testset[1]).reshape(-1)
    y_pre = svc.predict(X)
    print(svc.score(X, test_label))

    # from sklearn.metrics import cohen_kappa_score
    # print('kappa', cohen_kappa_score(test_label, y_pre))

def classifier_by_svm_without_decision(X_trainset, y_trainset, X_testset, y_testset, n_views, gamma):
    X = np.hstack((X_trainset[i]) for i in range(n_views)).T
    label = np.hstack((y_trainset[i]) for i in range(n_views)).T

    from sklearn.svm import SVC
    svc = SVC(kernel='rbf', gamma=gamma)
    svc.fit(X, label)

    X = np.hstack((X_testset[i]) for i in range(n_views)).T
    label = np.hstack((y_testset[i]) for i in range(n_views)).T

    print('OA', svc.score(X, label))
    y_pre = svc.predict(X)
    from sklearn.metrics import cohen_kappa_score
    print('kappa', cohen_kappa_score(label, y_pre))

    return y_pre

def experiment_on_svm(dataset, n_components, gamma):
    if dataset == 'indian':
        X_spectral, Y = IndianDatasetProcess.getIndianDataset('NULL')
    elif dataset == 'salina':
        X_spectral, Y = SalinaDatasetProcess.getSalinaDataset('NULL')

    sc = StandardScaler()
    X_spectral = sc.fit_transform(X_spectral.T).T

    X_multi = [X_spectral]
    Y_multi = [Y]

    # print('step2')
    n_views = len(X_multi)
    X_multi = PCA_reduction(X_multi, n_components)

    # print('step3')
    X_trainset, X_testset, y_trainset, y_testset = split_dataset_multi(X_multi, Y_multi, 0.85)

    classifier_by_svm_without_decision(X_trainset, y_trainset, X_testset, y_testset, n_views, gamma)

def experiment(algorithm, dataset, spatialFeature, n_components, gamma):
    '''
    :param algorithm: string ['MvDA' | 'GMMFA' | 'GMMDP']
        algorithm for multiview learning
    :param dataset: string ['indian' | 'salina'] 
        hyperspectral images dataset, Indian Pines dataset or Salinas Valley dataset
    :param spatialFeature: string ['neighbor' | 'wavelet']
        method for spatial feature extraction
    :param n_components: ingter
    :return: void
    '''
    # print('step1')
    if dataset == 'indian':
        X_spectral, X_spatial, Y = IndianDatasetProcess.getIndianDataset(spatialFeature)
    elif dataset == 'salina':
        X_spectral, X_spatial, Y = SalinaDatasetProcess.getSalinaDataset(spatialFeature)

    sc = StandardScaler()
    X_spectral = sc.fit_transform(X_spectral.T).T
    X_spatial = sc.fit_transform(X_spatial.T).T
    X_multi = [X_spectral, X_spatial]
    Y_multi = [Y, Y]

    # print('step2')
    n_views = len(X_multi)
    X_multi = PCA_reduction(X_multi, n_components)

    # print('step3')
    X_trainset, X_testset, y_trainset, y_testset = split_dataset_multi(X_multi, Y_multi, 0.6)

    # print('step4')
    if algorithm == 'MvDA':
        import MvDA
        W = MvDA.MvDA(X_trainset, y_trainset, n_components)
    elif algorithm == 'GMMFA':
        import GMMFA
        W = GMMFA.GMMFA(X_trainset, y_trainset, n_components)
    elif algorithm == 'GMMDP':
        import GMMDP
        W = GMMDP.GMMDP(X_trainset, y_trainset, 1, 10, n_components)

    # print('step5')
    X_trainset = projecting(X_trainset, W)
    X_testset = projecting(X_testset, W)

    # print('step6')
    classifier_by_svm(X_trainset, y_trainset, X_testset, y_testset, n_views, gamma)
    # print('--------------------------------------------------------')
    # classifier_by_svm_without_decision(X_trainset, y_trainset, X_testset, y_testset, n_views, gamma)

import matplotlib.pyplot as plt

def showDiagram():
    lda_acc = [0.6135,0.6133,0.6187,0.6214]
    svm_acc = [0.7626,0.7934,0.8084,0.8178]
    MvDA_acc = [0.946740438,0.962083897,0.970284477,0.97661494]
    GMMFA_acc = [0.927110897,0.947361299,0.956400742,0.965896788163]
    GRMMDP_acc = [0.957324994,0.974370771,0.981292517,0.985113677]
    x = [0.1000,0.2000,0.3000,0.4000]
    plt.figure()
    plt.title("Evluation on Indian Pines dataset with trainset size varying")
    plt.plot(x, lda_acc, "k--", label='LDA-MLE')
    plt.plot(x, svm_acc, 'g:', label='SVM-RBF')
    plt.plot(x, MvDA_acc, 'sy-.', label='MvDA')
    plt.plot(x, GMMFA_acc, 'mo-', label='GMMFA')
    plt.plot(x, GRMMDP_acc, 'r^-', label='GMMDP')
    plt.xlabel("trainset size")
    plt.ylabel("Mean Accuracy")
    plt.legend(loc='best')
    plt.show()

# print('MvDA', 'indian', 'neighbor')
# for i in range(20):
#     experiment('MvDA', 'indian', 'neighbor', 20, 0.45)
#
# print('MvDA', 'indian', 'wavelet')
# for i in range(20):
#     experiment('MvDA', 'indian', 'wavelet', 20, 0.45)
#
# print('GMMDP', 'indian', 'neighbor')
# for i in range(20):
#     experiment('GMMDP', 'indian', 'neighbor', 20, 1.2)
#
# print('GMMDP', 'indian', 'wavelet')
# for i in range(20):
#     experiment('GMMDP', 'indian', 'neighbor', 20, 1.2)
#
# print('GMMFA', 'indian', 'neighbor')
# for i in range(20):
#     experiment('GMMFA', 'indian', 'neighbor', 20, 0.1)
#
# print('GMMFA', 'indian', 'wavelet')
# for i in range(20):
#     experiment('GMMFA', 'indian', 'wavelet', 20, 0.1)

# print('MvDA', 'salina', 'neighbor')
# for i in range(20):
#     experiment('MvDA', 'salina', 'neighbor', 20, 5)

# print('GMMDP', 'salina', 'neighbor')
# for i in range(20):
#     experiment('GMMDP', 'salina', 'neighbor', 20, 15)

# print('GMMFA', 'salina', 'neighbor')
# for i in range(20):
#     experiment('GMMFA', 'salina', 'neighbor', 20, 5)

# print('MvDA', 'salina', 'wavelet')
# for i in range(20):
#     experiment('MvDA', 'salina', 'wavelet', 20, 5)

# print('GMMDP', 'salina', 'wavelet')
# for i in range(20):
#     experiment('GMMDP', 'salina', 'wavelet', 20, 5)

# print('GMMFA', 'salina', 'wavelet')
# for i in range(20):
#     experiment('GMMFA', 'salina', 'wavelet', 20, 0.1)

# showDiagram()
