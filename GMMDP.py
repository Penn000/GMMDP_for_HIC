import numpy as np

def MDP(X, y, weighting):
    X = np.array(X)
    (n_features, n_samples) = X.shape

    n_classes = len(np.unique(y))
    avg_dis = [0] * n_classes
    book = [0] * n_classes

    # computing pairwise similaity
    dist = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i, n_samples):
            dist[j][i] = dist[i][j] = np.linalg.norm(np.array(X[:, i]) - np.array(X[:, j]))
            if y[i] == y[j]:
                avg_dis[y[i]] += dist[i][j]
                book[y[i]] += 1
    for i in range(n_classes):
        if book[i] == 0:
            print(avg_dis[i])
        avg_dis[i] /= book[i]

    # computing affinity matrix
    W1 = np.zeros((n_samples, n_samples))  # intra-class affinity matrix
    W2 = np.zeros((n_samples, n_samples))  # inter-class affinity matrix

    # computing maximum intra-class distance
    for i in range(n_samples):
        id = [j for j in range(n_samples) if y[i] == y[j] and i != j]
        maxid = id[0]
        for j in id:
            if dist[i][j] > dist[i][maxid]:
                maxid = j
        W1[i][maxid] = W1[maxid][i] = 1

    # computing minimum inter-class distance
    for i in range(n_samples):
        id = [j for j in range(n_samples) if y[i] != y[j]]
        minid = id[0]
        for j in id:
            if dist[i][j] < dist[i][minid]:
                minid = j
        W2[i][minid] = W2[minid][i] = 1

    # intra-class Laplacian matrix L_intra
    # inter-class Laplacian matrix L_inter
    L1 = np.diag([sum(W1[i]) for i in range(n_samples)]) - W1
    L2 = np.diag([sum(W2[i]) for i in range(n_samples)]) - W2

    L1 = 0.5 * (L1 + L1.T)
    L2 = 0.5 * (L2 + L2.T)
    Sb = X.dot(L2).dot(X.T)
    Sw = X.dot(L1).dot(X.T)

    W = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        id = [j for j in range(n_samples) if y[i] == y[j] and i != j]
        for j in id:
            if weighting == 0:  # 0-1 weighting
                W[i][j] = 1
            elif weighting == 1:  # heat kernel weighting
                W[i][j] = np.exp(-dist[i][j] / avg_dis[y[i]])
            else:  # dot-product weighting
                W[i][j] = np.array(X[:, i]).T.dot(np.array(X[:, j]))

    Dfallten = [sum(W[i]) for i in range(n_samples)]
    D = np.diag(Dfallten)
    L = D - W
    G = X.dot(L).dot(X.T)
    G = 0.5 * (G + G.T)

    # print('MDP finish')
    return Sb, Sw, G

# 参数列表：coefficient, weighting,
def GMMDP(X_multiview, label_multiview, weighting, Lambda, n_components):
    n_views = len(X_multiview)  # 视图个数

    coefficient = np.array([
        [1, 10, 10, 10, 10],
        [10, 1, 10, 10, 10],
        [10, 10, 1, 10, 10],
        [10, 10, 10, 1, 10],
        [10, 10, 10, 10, 1]
    ])
    '''
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]

        [1, 5, 5, 5, 5],
        [5, 1, 5, 5, 5],
        [5, 5, 1, 5, 5],
        [5, 5, 5, 1, 5],
        [5, 5, 5, 5, 1]

        [1, 10, 10, 10, 10],
        [10, 1, 10, 10, 10],
        [10, 10, 1, 10, 10],
        [10, 10, 10, 1, 10],
        [10, 10, 10, 10, 1]
    '''

    n_samples = [0] * n_views
    n_features = [0] * n_views
    for v in range(n_views):
        X_multiview[v] = np.array(X_multiview[v])
        n_samples[v] = X_multiview[v].shape[1]  # 样本数
        n_features[v] = X_multiview[v].shape[0]  # 样本特征数，即维数

    B = [0] * n_views
    A = [0] * n_views
    for vi in range(n_views):
        sb, sw, G = MDP(X_multiview[vi], label_multiview[vi], weighting)
        A[vi] = sb
        B[vi] = sw + Lambda * G
    A = np.array(A)
    B = np.array(B)

    AA = np.ones((sum(n_features), sum(n_features)))
    for i in range(n_views):
        AA[sum(n_features[: i]): sum(n_features[:i + 1]), sum(n_features[:i]): sum(n_features[:i + 1])] = \
            coefficient[i][i] * A[i]
    for i in range(n_views):
        for j in range(i + 1, n_views):
            AA[sum(n_features[: i]): sum(n_features[: i + 1]), sum(n_features[: j]): sum(n_features[: j + 1])] = \
                coefficient[i][j] * X_multiview[i].dot(X_multiview[j].T)
            AA[sum(n_features[: j]): sum(n_features[: j + 1]), sum(n_features[: i]): sum(n_features[: i + 1])] = \
                coefficient[i][j] * X_multiview[j].dot(X_multiview[i].T)

    BB = np.zeros((sum(n_features), sum(n_features)))
    for i in range(n_views):
        BB[sum(n_features[: i]): sum(n_features[:i + 1]), sum(n_features[:i]): sum(n_features[:i + 1])] = B[i]

    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(BB).dot(AA))

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
    # print('MvMDP finish')
    return W