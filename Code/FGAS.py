# Fuzzy granular anomaly detection using Markov random walk (FGAS) algorithm
# Please refer to the following papers:
# Fuzzy granular anomaly detection using Markov random walk, Information Sciences, 2023.
# Uploaded by Yuan Zhong on Aug. 6, 2023. E-mail:yuanzhong2799@foxmail.com.
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler


def FGAS(data, sigma):
    # input:
    # data is data matrix without decisions, where rows for samples and columns for attributes.
    # All attributes should be normalized into [0,1]
    # sigma is a given parameter for the radius adjustment
    # output
    # Fuzzy anomaly score AS.

    d = 0.1
    n, m = data.shape
    phi = np.zeros(n)
    Dis = np.zeros((n, n))
    for j in range(m):
        sim = 1 - squareform(pdist(data[:, j].reshape(-1, 1), 'cityblock'))
        sim[sim < sigma] = 0
        temp = sim.sum(axis=1) / n
        Dis += squareform(pdist(temp.reshape(-1, 1)))

    A = Dis
    diag_A = A.sum(axis=1)
    B = np.diag(diag_A)
    # P = np.linalg.inv(B) @ A
    P = np.linalg.solve(B, A)

    pi_t = np.ones(n) / n
    pi_t_temp = np.ones(n)
    i = 0
    while np.linalg.norm(pi_t_temp - pi_t, 1) > 0.0001:
        pi_t_temp = pi_t
        pi_t = d + (1 - d) * pi_t @ P
        i += 1

    pi_t_w = pi_t
    phi[:n] = (pi_t_w - min(pi_t_w)) / (max(pi_t_w) - min(pi_t_w))

    # Fuzzy anomaly score AS.
    AS = phi
    return AS

if __name__ == "__main__":
    load_data = loadmat('Example.mat')
    trandata = load_data['Example']
    scaler = MinMaxScaler()
    trandata = scaler.fit_transform(trandata)

    sigma = 0.6
    anomaly_scores = FGAS(trandata, sigma)
    print(anomaly_scores)