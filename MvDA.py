import numpy as np

# input: X(一个视图，每一列表示一个样本, d×m, m为样本数，d为维数)
#        y(标签向量， 1×m, m为样本数)
# output: u_mean(表示第i类的均值向量)
#         n_num(表示第i类的样本个数)
#         sample_of_class(第i类的样本)
def get_mean_view(X, y):
    X= np.array(X).T
    class_labels = np.unique(y)
    n_classes = class_labels.shape[0]
    n_sample = X.shape[0]
    n_features = X.shape[1]

    # 计算第i类的样本个数
    n_num = [0]*(n_classes+1)
    sample_of_class = {}
    for i in range(n_classes):
        sample_of_class[i] = []
    for i in range(n_sample):
        n_num[y[i]] += 1
        sample_of_class[y[i]].append(X[i])
    for i in range(n_classes):
        sample_of_class[i] = np.array(sample_of_class[i])

    # 计算第i类的均值向量
    u_mean = np.zeros((n_classes, n_features))
    for i in range(n_sample):
        u_mean[y[i]] += X[i]
    for cl in class_labels:
        u_mean[cl] /= n_num[cl]
    return u_mean, n_num, sample_of_class

# input: X_multiview(v个视图，每个视图为一个矩阵，矩阵每一列表示一个样本)
#        Label_multiview（v个视图的标签）
#        n_components(降到的维数)
# output:W(投影矩阵)
def MvDA(X_multiview, y_multiview, n_components):
    n_view = len(X_multiview)  # 视图个数
    n_class = len(np.unique(y_multiview[0]))  # 种类个数

    n_features = [0] * n_view
    for v in range(n_view):
        n_features[v] = len(X_multiview[v])  # 样本特征数，即维数

    for vi in range(n_view):
        X_multiview[vi] = np.array(X_multiview[vi])

    # ####################__Sjr__#################################
    Ni = np.zeros(n_class)  # 每个种类的样本数
    A = {}
    for vi in range(n_view):
        u, n_num, sample_class = get_mean_view(X_multiview[vi], y_multiview[vi])
        A[vi] = [u, n_num, sample_class]
        for i in range(n_class):
            Ni[i] += n_num[i]

    # ####################__LDA Sw__##############################
    Sw = np.zeros((sum(n_features), sum(n_features)))
    for i in range(n_view):
        Numi = A[i][1]
        Mi = A[i][0]
        for j in range(i, n_view):
            Numj = A[j][1]
            Mj = A[j][0]
            Xj = A[j][2]
            sij = np.zeros((n_features[i], n_features[j]))
            vij = np.zeros((n_features[j], n_features[j]))
            for ci in range(n_class):
                Mici = Mi[ci].reshape(n_features[i], 1)
                Mjci = Mj[ci].reshape(n_features[j], 1)
                sij = sij - (Numi[ci] * Numj[ci] / Ni[ci]) * (Mici.dot(Mjci.T))
                for k in range(Numj[ci]):
                    Xijk = Xj[ci][k].reshape(n_features[j], 1)
                    vij += Xijk.dot(Xijk.T)
            if j == i:
                sij = vij + sij

            Sw[sum(n_features[: i]): sum(n_features[:i+1]), sum(n_features[:j]): sum(n_features[:j+1])] = sij
            Sw[sum(n_features[: j]): sum(n_features[:j+1]), sum(n_features[:i]): sum(n_features[:i+1])] = sij.T

    # ####################__LDA Sb__##############################
    Sb = np.zeros((sum(n_features), sum(n_features)))
    n = sum(Ni)

    for i in range(n_view):
        mi = np.sum(X_multiview[i], axis=1).reshape(n_features[i], 1)
        Numi = A[i][1]
        Mi = A[i][0]
        for j in range(i, n_view):
            Numj = A[j][1]
            Mj = A[j][0]
            sij = np.zeros((n_features[i], n_features[j]))
            mj = np.sum(X_multiview[j], axis=1).reshape(n_features[j], 1)
            for ci in range(n_class):
                Mici = Mi[ci].reshape(n_features[i], 1)
                Mjci = Mj[ci].reshape(n_features[j], 1)
                sij = sij + (Numi[ci] * Numj[ci] / Ni[ci]) * (Mici.dot(Mjci.T))

            sij = sij - mi.dot(mj.T)/n
            Sb[sum(n_features[: i]): sum(n_features[:i+1]), sum(n_features[:j]): sum(n_features[:j+1])] = sij
            Sb[sum(n_features[: j]): sum(n_features[:j+1]), sum(n_features[:i]): sum(n_features[:i+1])] = sij.T

    # LDA
    Sb = Sb * n_view

    SwSb = np.linalg.inv(Sw).dot(Sb)
    eig_vals, eig_vecs = np.linalg.eig(SwSb)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i].real) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    eig_sorted = np.hstack((eig_pairs[i][1].reshape(sum(n_features), 1)) for i in range(n_components))
    # print('eig_sorted:', eig_sorted.shape)

    W = [[]] * n_view
    for i in range(n_view):
        W[i] = eig_sorted[sum(n_features[:i]):sum(n_features[:i + 1]), :]

    W = np.array(W)

    # print('MvDA finished')
    return W