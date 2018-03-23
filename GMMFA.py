import numpy as np

# Marginal Fisher Analysis
# defualt k = 1
#
# input: X(d*m) 数据矩阵，一列表示一个样本
#        label(1*m) 标记向量
#        n_components 降到的维数
#
# output: A, B(投影矩阵)
def MFA(X, label):
    X = np.array(X)
    n_features = X.shape[0]
    n_samples = X.shape[1]

    k1 = 15
    k2 = 5

    dist = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i, n_samples):
            dist[j][i] = dist[i][j] = np.linalg.norm(np.array(X[:, i])-np.array(X[:, j]))

    N1 = [0]*n_samples
    N2 = [0]*n_samples
    for i in range(n_samples):
        tmp = [(dist[i][j], j) for j in range(n_samples)]
        tmp.sort(key=lambda x:x[0], reverse=False)
        n1 = []
        n2 = []
        cnt1 = 0
        cnt2 = 0
        for j in range(n_samples):
            if i == tmp[j][1]:
                continue
            if label[i] == label[tmp[j][1]] and cnt1 < k1:
                n1.append(tmp[j][1])
                cnt1 += 1
            if label[i] != label[tmp[j][1]] and cnt2 < k2:
                n2.append(tmp[j][1])
                cnt2 += 1
            if cnt1>=k1 and cnt2>=k2:
                break
        N1[i] = n1
        N2[i] = n2

    W1 = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            if i in N1[j] or j in N1[i]:
                W1[i][j] = 1
                W1[j][i] = 1

    W2 = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            if i in N2[j] or j in N2[i]:
                W2[i][j] = 1
                W2[j][i] = 1

    L1 = np.diag([sum(W1[i]) for i in range(n_samples)]) - W1
    L2 = np.diag([sum(W2[i]) for i in range(n_samples)]) - W2

    Sb = X.dot(L2).dot(X.T)
    Sw = X.dot(L1).dot(X.T)

    # print('MFA finish')
    return Sb, Sw

# Generalized Multi-view Marginal Fisher Analysis
# defualt k = 1
# note: 每个视图的样本数需一样，且视图间的样本要一一对应
#
# input: X_multiview(v*(d*m)) 数据矩阵，一列表示一个样本
#        label_multiview(v*(1*m)) 标记向量
#        n_components 降到的维数
#
# output: X_transform(n_components*m) 降维后的数据矩阵
def GMMFA(X_multiview, label_multiview, n_components):
    n_views = len(X_multiview)  # 视图个数
    n_classes = len(np.unique(label_multiview[0]))  # 种类个数

    coefficient = np.array([
        [1, 10],
        [10, 1]
    ])

    n_features = [0] * n_views
    for v in range(n_views):
        n_features[v] = len(X_multiview[v])  # 样本特征数，即维数

    X_multiview = np.array(X_multiview)
    for vi in range(n_views):
        X_multiview[vi] = np.array(X_multiview[vi])

    A = [0]*n_views
    B = [0]*n_views
    for vi in range(n_views):
        a, b = MFA(X_multiview[vi], label_multiview[vi])
        A[vi] = a
        B[vi] = b
    A = np.array(A)
    B = np.array(B)

    AA = np.zeros((sum(n_features), sum(n_features)))
    for i in range(n_views):
        a = coefficient[i][i]*A[i]
        AA[sum(n_features[: i]): sum(n_features[:i + 1]), sum(n_features[:i]): sum(n_features[:i + 1])] = a
    for i in range(n_views):
        for j in range(i+1, n_views):
            AA[sum(n_features[: i]): sum(n_features[:i + 1]), sum(n_features[:j]): sum(n_features[:j + 1])] = \
            coefficient[i][j]*X_multiview[i].dot(X_multiview[j].T)
            AA[sum(n_features[: j]): sum(n_features[:j + 1]), sum(n_features[:i]): sum(n_features[:i + 1])] = \
            coefficient[j][i] * X_multiview[j].dot(X_multiview[i].T)

    gama = np.zeros(n_views)
    gama[0] = 1
    for i in range(1, n_views):
        gama[i] = (np.trace(B[i-1]))/(np.trace(B[i]))
    BB = np.zeros((sum(n_features), sum(n_features)))
    for i in range(n_views):
        b = gama[i]*B[i]
        BB[sum(n_features[: i]): sum(n_features[:i + 1]), sum(n_features[:i]): sum(n_features[:i + 1])] = b

    SwSb = np.linalg.inv(BB).dot(AA)
    eig_vals, eig_vecs = np.linalg.eig(SwSb)

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i].real) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

    eig_sorted = np.hstack((eig_pairs[i][1].reshape(sum(n_features), 1)) for i in range(n_components))
    # print('eig_sorted:', eig_sorted.shape)

    W = [[]] * n_views
    for i in range(n_views):
        W[i] = eig_sorted[sum(n_features[:i]):sum(n_features[:i + 1]), :]

    W = np.array(W)
    # print('GMMFA finish')

    return W