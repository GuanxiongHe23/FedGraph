import torch
import numbers
import numpy as np
import torch.nn as nn
import scipy.io as scio
from sklearn.cluster import KMeans,SpectralClustering
from skfuzzy import cmeans
from scipy.stats import gaussian_kde
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.stats import gaussian_kde

from matplotlib import pyplot as plt
from scipy import io
import itertools
from sklearn import metrics

from sklearn.mixture import GaussianMixture
import skfuzzy as fuzz
import os

import pandas as pd
import math
from sklearn.metrics.pairwise import cosine_similarity

import pickle as pkl


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
# import optim
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import MinMaxScaler

device = "cuda:0"
eps = 2.2204e-16

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    s = 0
    for i, j in zip(row_ind, col_ind):
        s += w[i, j]
    return s * 1.0 / y_pred.size

def L2_distance_1(a, b):
    # Compute the squared Euclidean distance between all pairs of vectors in a and b
    aa = torch.sum(a * a, dim=0)
    bb = torch.sum(b * b, dim=0)
    ab = torch.matmul(a.T, b)
    d = aa[:, None] + bb[None, :] - 2 * ab
    d = torch.clamp(d, min=0)  # Ensure no negative distances
    return d.to(device)

def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result

def cal_weights_via_CAN(X, num_neighbors, links=0):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return:
    """
    size = X.shape[1]
    distances = L2_distance_1(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 10**-10

    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))
    sorted_distances = None
    torch.cuda.empty_cache()
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().to(device)
    if links is not 0:
        links = torch.Tensor(links).to(device)
        weights += torch.eye(size).to(device)
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.cuda().to(device)
    weights = weights.cuda().to(device)
    
    return weights, raw_weights

def cal_similarity_matrix_dist(X):
    """
    Calculate similarity matrix based on Euclidean distance
    :param X: d * n
    :return: n * n
    """
    n = X.shape[1]
    S = distance(X, X)
    S = S - torch.diag(torch.diag(S))
    S = S.relu()
    S = S / S.max()
    return S

def eig1(L, c, isMax):
    # Using torch.symeig to find eigenvalues and eigenvectors
    L = torch.tensor(L, dtype=torch.float32).to(device)
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    if isMax:
        indices = torch.argsort(eigenvalues, descending=True)
    else:
        indices = torch.argsort(eigenvalues)
    selected_eigenvalues = eigenvalues[indices][:c]
    selected_eigenvectors = eigenvectors[:, indices][:, :c]
    return selected_eigenvectors, selected_eigenvalues, eigenvalues


def EProjSimplex_new(v, k=1):
    # Euclidean Projection onto the Simplex
    v = torch.tensor(v, dtype=torch.float32).to(device)
    u, _ = torch.sort(v, descending=True)
    sv = torch.cumsum(u, dim=0).to(device)
    rho = torch.nonzero(u * torch.arange(1, len(v) + 1).to(device) > (sv - k), as_tuple=False).max()
    theta = (sv[rho] - k) / (rho + 1.0)
    w = torch.clamp(v - theta, min=0)
    return w

def EProjSimplexdiag(d, u):
    """
    Solve the problem:
        min  1/2*x'*U*x - x'*d
        s.t. x>=0, 1'x=1
    """
    
    lambda_ = torch.min(u - d)
    f = 1
    count = 1
    
    while abs(f) > 1e-8:
        v1 = 1.0 / u * lambda_ + d / u
        posidx = v1 > 0
        g = torch.sum(1.0 / u[posidx])
        f = torch.sum(v1[posidx]) - 1
        lambda_ = lambda_ - f / g
        
        if count > 1000:
            break
        count += 1
    
    v1 = 1.0 / u * lambda_ + d / u
    v1[v1 < 0] = 0
    x = v1
    
    return x, f

class Loss_Cluster(nn.Module):
    def __init__(self):
        super(Loss_Cluster, self).__init__()

    def forward(self, A, S):
        diff = A - S
        frobenius_norm = torch.norm(diff, p='fro')
        loss = frobenius_norm ** 2
        return loss

def CLR(A0, c, isrobust=1, islocal=1, device=device):
    NITER = 30
    zr = 10e-11
    lambda_ = 0.01
    r = 0
    
    A0 = A0 - torch.diag(torch.diag(A0))
    num = A0.shape[0]
    A10 = (A0 + A0.T) / 2
    D10 = torch.diag(torch.sum(A10, dim=1))
    L0 = D10 - A10
    F0, _, evs = eig1(L0, num, 0)
    
    a = torch.abs(evs)
    a[a < zr] = eps
    ad = torch.diff(a)
    ad1 = ad / a[1:]
    ad1[ad1 > 0.85] = 1
    ad1 = ad1 + eps * torch.arange(1, num, dtype=torch.float32).to(device)
    ad1[0] = 0
    ad1 = ad1[:int(0.9 * len(ad1))]
    
    te, cs = torch.sort(ad1, descending=True)
    cs = cs
    print('Suggested cluster number is: %d, %d, %d, %d, %d' % (cs[0], cs[1], cs[2], cs[3], cs[4]))
    if c is None:
        c = cs[0]
    
    F = F0[:, :c]
    u = {}
    
    if torch.sum(evs[:c + 1]) < zr:
        raise ValueError(f'The original graph has more than {c} connected components')
    
    if torch.sum(evs[:c]) < zr:
        clusternum, y = connected_components(A10.cpu(), directed=False)
        S = A0
        return y, S, evs, cs
    
    for i in range(num):
        a0 = A0[i, :]
        if islocal == 1:
            idxa0 = torch.where(a0 > 0)[0].to(device)
        else:
            idxa0 = torch.arange(num)
        u[i] = torch.ones(len(idxa0)).to(device)
    
    for iter in range(NITER):
        dist = L2_distance_1(F.T, F.T)
        S = torch.zeros((num, num), dtype=torch.float32).to(device)
        for i in range(num):
            a0 = A0[i, :]
            if islocal == 1:
                idxa0 = torch.where(a0 > 0)[0]
            else:
                idxa0 = torch.arange(num)
            ai = a0[idxa0]
            di = dist[i, idxa0]
            
            if isrobust == 1:
                ad = u[i] * ai - lambda_ * di / 2
                si,_ = EProjSimplexdiag(ad, u[i] + r * torch.ones(len(idxa0)).to(device))
                u[i] = 1 / (2 * torch.sqrt((si - ai) ** 2 + eps))
                S[i, idxa0] = torch.tensor(si, dtype=torch.float32).to(device)
            else:
                ad = ai - 0.5 * lambda_ * di
                S[i, idxa0] = EProjSimplex_new(ad)
                
        A = S
        A = (A + A.T) / 2
        D = torch.diag(torch.sum(A, dim=-1))
        L = D - A
        F_old = F
        F, _, ev = eig1(L, c, 0)
        evs = torch.vstack((evs, ev))
        
        fn1 = torch.sum(ev[:c])
        fn2 = torch.sum(ev[:c + 1])
        
        if fn1 > zr:
            lambda_ *= 2
        elif fn2 < zr:
            lambda_ /= 2
            F = F_old
        else:
            break
    
    clusternum, y = connected_components(csr_matrix(A.cpu()), directed=False)
    if clusternum != c:
        print(f'Cannot find the correct cluster number: {c}')

    return y, S, evs, cs


class Encoder_Lin(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder_Lin, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder_Lin(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder_Lin, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )
        nn.init.eye_(self.decoder[0].weight)
        nn.init.zeros_(self.decoder[0].bias)

    def forward(self, x):
        return self.decoder(x)

class ClusteringLayer(nn.Module):
    def __init__(self, num_clusters, num_features, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.num_clusters = num_clusters
        self.num_features = num_features
        self.alpha = alpha
        self.mu = nn.Parameter(torch.Tensor(num_clusters, num_features).to(device))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mu.data)

    def set_parameters(self, mu):
        self.mu.data = mu
    
    def forward(self, x):
        q = 1.0 / (1.0 + (torch.sum((x.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha))
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

class ClusteringCLR(nn.Module):
    def __init__(self, cluster_num):
        self.cluster_num = cluster_num
    
    def fit(self, A):
        y, S, evs, cs = CLR(A, self.cluster_num)
        self.S = S
        self._labels = y
        return y, S
    
    def get_labels(self):
        return self._labels


class Loss_GraphDiff(nn.Module):
    def __init__(self):
        super(Loss_GraphDiff, self).__init__()

    def forward(self, A, S):
        # Calculate the difference between the two matrices
        diff = A - S
        # Compute the Frobenius norm of the difference
        frobenius_norm = torch.norm(diff, p='fro')
        # Return the square of the Frobenius norm
        loss = frobenius_norm ** 2
        return loss

def constructW_PKN(X, k=5, issymmetric=True):
    """
    Construct similarity matrix with probabilistic k-nearest neighbors.
    Args:
        X (torch.Tensor): Data matrix, each column is a data point
        k (int): Number of neighbors
        issymmetric (bool): If True, set W = (W + W.t()) / 2
    Returns:
        torch.Tensor: Similarity matrix W
    """
    dim, n = X.shape
    D = L2_distance_1(X, X)
    _, idx = torch.sort(D, dim=1)  # sort each row

    W = torch.zeros(n, n).to(device)
    for i in range(n):
        id = idx[i, 1:k+2]
        di = D[i, id]
        W[i, id] = (di[k] - di) / (k * di[k] - torch.sum(di[:k]) + eps)

    if issymmetric:
        W = (W + W.t()) / 2

    return W


class LocalModel(nn.Module):
    def __init__(self, num_clusters, num_neighbors, in_dim, links=10):
        super(LocalModel, self).__init__()
        self.num_clusters = num_clusters
        self.num_neighbors = num_neighbors
        self.links = links
        
        self.encoder = Decoder_Lin(in_dim, 2*in_dim, in_dim).to(device)
    
    def init_params(self, X):
        self.encoder.apply(self.init_weights)
        
    def forward(self, X):
        embedding = self.encoder(X)
        #self.A = constructW_PKN(embedding.T, self.num_neighbors)
        self.A, raw_weights = cal_weights_via_CAN(embedding.T, self.num_neighbors)
        return embedding, self.A
    
    def train(self, X, S, num_epochs, lr, momentum, weight_decay, init=True):
        if init:
            self.init_params(X)
        
        loss_dif = Loss_GraphDiff()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            embedding, GraphA = self.forward(X)
            
            loss = loss_dif(GraphA, S)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
        
    
    def parameters(self):
        return list(self.encoder.parameters())


# 2. PrototypeGenerator class with modified sensitivity calculation
class PrototypeGenerator:
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        
    def compute_prototypes(self, data):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)  # Increase the number of initializations
        kmeans.fit(data)
        self.prototypes = kmeans.cluster_centers_
        return self.prototypes

proto_gen = PrototypeGenerator()

def calculate_sensitivity(data_samples_list, n_clusters=10):
    """
    Calculate the sensitivity of cluster centers.
    
    Parameters:
        data_samples_list (list of np.ndarray): List of data from all clients.
        n_clusters (int): Number of clusters.
    
    Returns:
        sensitivity (float): The sensitivity.
    """
    sensitivities = []
    
    for data_samples in data_samples_list:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(data_samples)
        prototypes = kmeans.cluster_centers_
        
        # Simulate the change after removing one sample
        idx = np.random.randint(0, data_samples.shape[0])
        perturbed_data = np.delete(data_samples, idx, axis=0)
        
        kmeans_perturbed = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans_perturbed.fit(perturbed_data)
        perturbed_prototypes = kmeans_perturbed.cluster_centers_
        
        delta = np.max(np.linalg.norm(prototypes - perturbed_prototypes, axis=1))
        sensitivities.append(delta)
    
    return np.max(sensitivities)

def allocate_privacy_budget(class_counter, total_epsilon, min_ratio=1/5):
    """
    Dynamically allocate privacy budget based on the number of samples in each class, 
    ensuring the budget for each class is not below a minimum limit.
    
    Parameters:
        class_counter (list or np.ndarray): List of sample counts for each class.
        total_epsilon (float): Total privacy budget.
        min_ratio (float): The minimum ratio of a single class's budget to the total budget, default is 1/5.
    
    Returns:
        epsilon_per_class (np.ndarray): Array of privacy budgets allocated to each class.
    """
    class_counter = np.asarray(class_counter)
    n_classes = len(class_counter)
    
    # Weights are positively correlated with the number of samples
    weights = class_counter.astype(float)
    normalized_weights = weights / np.sum(weights)
    
    # Initial allocation
    epsilon_per_class = normalized_weights * total_epsilon
    
    # Calculate the minimum budget
    min_epsilon = total_epsilon * min_ratio / n_classes
    
    # Apply the minimum budget constraint
    if np.any(epsilon_per_class < min_epsilon):
        low_indices = epsilon_per_class < min_epsilon
        epsilon_per_class[low_indices] = min_epsilon
        remaining_budget = total_epsilon - np.sum(epsilon_per_class)
        valid_indices = ~low_indices
        
        if np.sum(valid_indices) == 0:
            # All classes require the minimum budget, handle the excess
            total_allocated = np.sum(epsilon_per_class)
            if total_allocated > total_epsilon:
                excess = total_allocated - total_epsilon
                epsilon_per_class -= excess / n_classes
        else:
            # Allocate the remaining budget
            sum_valid_weights = np.sum(normalized_weights[valid_indices])
            epsilon_per_class[valid_indices] += remaining_budget * (
                normalized_weights[valid_indices] / sum_valid_weights
            )
    
    return epsilon_per_class

def laplace_noise_addition(true_values, sensitivity, epsilon):
    """
    Add noise independently to each dimension.
    
    Parameters:
        true_values (np.ndarray): Original values, shape is (n_classes, n_features).
        sensitivity (float): The sensitivity.
        epsilon (np.ndarray): Privacy budget for each class, shape is (n_classes,).
    
    Returns:
        noisy_values (np.ndarray): Noisy values, shape is (n_classes, n_features).
    """
    n_classes, n_features = true_values.shape
    
    # Initialize noisy result
    noisy_values = np.zeros_like(true_values)
    
    scale_rec = []
    for i in range(n_classes):
        scale = sensitivity / (epsilon[i] * n_features)  # Allocate privacy budget per dimension
        noise = np.random.laplace(loc=0, scale=scale, size=n_features)
        noisy_values[i] = true_values[i] + noise
        scale_rec.append(scale)
    
    print("Means Scale for each class:", scale_rec)
    
    # Modified constraint logic: ensure noisy values are within the [0, 1] range
    noisy_values = np.clip(noisy_values, a_min=0, a_max=1)
    
    return noisy_values


def add_noise_to_graph(graph, graph_sensitivity, epsilon_graph):
    """
    Add noise to the adjacency matrix.
    
    Parameters:
        graph (np.ndarray): Adjacency matrix, shape is (n_samples, n_samples).
        graph_sensitivity (float): Sensitivity of the adjacency matrix.
        epsilon_graph (float): Privacy budget for the adjacency matrix.
    
    Returns:
        noisy_graph (np.ndarray): Noisy adjacency matrix, shape is (n_samples, n_samples).
    """
    scale = graph_sensitivity / (epsilon_graph * graph.shape[0])  # Allocate privacy budget per dimension
    print("Graph scale:", scale)
    noise = np.random.laplace(loc=0, scale=scale, size=graph.shape)
    noisy_graph = graph + noise
    noisy_graph = np.clip(noisy_graph, a_min=0, a_max=1)  # Ensure adjacency matrix values are within the [0, 1] range
    return noisy_graph

def calculate_prototype_sensitivity(data_samples_list, n_clusters=10, num_removals=1, noise_scale=0.1):
    """
    Calculate the sensitivity of cluster centers.
    
    Parameters:
        data_samples_list (list of np.ndarray): List of data from all clients.
        n_clusters (int): Number of clusters.
        num_removals (int): Number of samples to simulate perturbation, default is 1.
        noise_scale (float): Scale of random perturbation (standard deviation), default is 0.1.
    
    Returns:
        sensitivity (float): The sensitivity.
    """
    sensitivities = []
    
    for data_samples in data_samples_list:
        # KMeans clustering on the original data
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(data_samples.T)  # Note: KMeans input is (n_samples, n_features), so a transpose is needed
        prototypes = kmeans.cluster_centers_
        
        # Randomly select indices of samples to perturb
        idx_to_perturb = np.random.choice(data_samples.shape[1], size=num_removals, replace=False)
        
        # Create a copy of the data and add random noise to the selected samples
        perturbed_data = data_samples.copy()
        noise = np.random.normal(loc=0, scale=noise_scale, size=(data_samples.shape[0], num_removals))
        perturbed_data[:, idx_to_perturb] += noise
        
        # Re-run KMeans clustering on the perturbed data
        kmeans_perturbed = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans_perturbed.fit(perturbed_data.T)
        perturbed_prototypes = kmeans_perturbed.cluster_centers_
        
        # Calculate the maximum change in cluster centers
        delta = np.max(np.linalg.norm(prototypes - perturbed_prototypes, axis=1))
        sensitivities.append(delta)
    
    return np.max(sensitivities)

def calculate_graph_sensitivity(data_samples_list, device, num_neighbors, num_removals=1, noise_scale=0.1):
    """
    Calculate the sensitivity of the adjacency matrix.
    
    Parameters:
        data_samples_list (list of np.ndarray): List of data from all clients.
        device (torch.device): Device (e.g., 'cuda:0' or 'cpu').
        num_neighbors (int): Number of neighbors used when constructing the adjacency matrix.
        num_removals (int): Number of samples to simulate perturbation, default is 1.
        noise_scale (float): Scale of random perturbation (standard deviation), default is 0.1.
    
    Returns:
        sensitivity (float): The sensitivity.
    """
    sensitivities = []
    
    for data_samples in data_samples_list:
        # Calculate the original adjacency matrix
        graph = cal_weights_via_CAN(torch.tensor(data_samples, dtype=torch.float32).to(device), num_neighbors)
        graph_np = graph[0].to('cpu').numpy()
        
        # Randomly select indices of samples to perturb
        idx_to_perturb = np.random.choice(data_samples.shape[1], size=num_removals, replace=False)
        
        # Create a copy of the data and add random noise to the selected samples
        perturbed_data = data_samples.copy()
        noise = np.random.normal(loc=0, scale=noise_scale, size=(data_samples.shape[0], num_removals))
        perturbed_data[:, idx_to_perturb] += noise
        
        # Calculate the perturbed adjacency matrix
        perturbed_graph = cal_weights_via_CAN(torch.tensor(perturbed_data, dtype=torch.float32).to(device), num_neighbors)
        perturbed_graph_np = perturbed_graph[0].to('cpu').numpy()
        
        # Calculate the maximum change in the adjacency matrix
        delta = np.max(np.abs(graph_np - perturbed_graph_np))
        sensitivities.append(delta)
    
    return np.max(sensitivities)


class Client:
    def __init__(self, data, num_clusters, num_neighbors, device):
        self.data = data
        self.num_clusters = num_clusters
        self.num_neighbors = num_neighbors
        self.n_samples = data.X.shape[1]
        self.init_ = True
        self.device = device
        
        self.encoder = LocalModel(num_clusters, self.num_neighbors, data.X.shape[1])

    def train(self, num_epochs = 200, lr = 0.01, momentum = 0.9, weight_decay = 0.0001):
        X = self.data.X
        y = self.data.y
        SX = self.data.SX
        SY = self.data.SY

        # Record the index mapping of shared and private data after merging the datasets
        shared_indices = range(X.shape[1], X.shape[1] + SX.shape[1])  # Indices of the shared data
        
        # Merge private and shared data
        X = torch.cat((X, SX), dim=1)  # Merge X and SX
        y = torch.cat((y, SY), dim=0)  # Merge y and SY
        
        self.sharing_sample_indices = []
        
        if self.init_:
            # self.gmm = GaussianMixture(n_components=self.num_clusters, covariance_type='full', n_init=20, init_params='random', random_state=42).fit(X.T.detach().cpu().numpy())
            # gmm_labels = torch.tensor(self.gmm.predict(X.T.detach().cpu().numpy())).to(self.device)
            # print('NMI GMM:', normalized_mutual_info_score(y.cpu().numpy(), gmm_labels.cpu().numpy()))
            
            gmm = SVC().fit(X.T.detach().cpu().numpy()[:int(X.shape[1] * 0.1),:], y.detach().cpu().numpy().squeeze()[:int(X.shape[1] * 0.1)])
            self.presudo_labels0 = torch.tensor(gmm.predict(X.T.detach().cpu().numpy())).to(self.device)
            print('acc:', accuracy_score(y.cpu().numpy(), self.presudo_labels0.cpu().numpy()))
            
            self.GraphA, weight_raw = cal_weights_via_CAN(torch.tensor(self.data.X, dtype=torch.float32).to(device), self.num_neighbors)
            self.presudo_labels, _, evs, cs = CLR(self.GraphA.to(device), self.num_clusters, 0, 0)
            self.presudo_labels = torch.tensor(self.presudo_labels).int().to(self.device)
            print('NMI CLR:', normalized_mutual_info_score(y.cpu().numpy(), self.presudo_labels.cpu().numpy()))
            
            reallocated_data = torch.zeros_like(X).to(self.device)
            reallocated_label = torch.zeros_like(y).to(self.device)
            reallocated_presudo = torch.zeros_like(self.presudo_labels).to(self.device)
            reallocated_presudo0 = torch.zeros_like(self.presudo_labels0).to(self.device)
            Prototype = []
            
            cnt_pos = 0
            #for i in np.random.choice(range(self.num_clusters), self.num_clusters, replace=False):
            for i in range(self.num_clusters):
                for j in range(torch.sum(y == i)):
                    
                    original_idx = torch.where(y == i)[0][j]
                    if original_idx in shared_indices:
                        self.sharing_sample_indices.append(cnt_pos)

                    reallocated_data[:, cnt_pos] = X[:, y == i][:, j]
                    reallocated_label[cnt_pos] = y[y == i][j]
                    reallocated_presudo[cnt_pos] = self.presudo_labels[y == i][j]
                    reallocated_presudo0[cnt_pos] = self.presudo_labels0[y == i][j]
                    cnt_pos += 1
                Prototype.append(torch.mean(X[:, y == i], dim=1))
            
            self.presudo_labels = reallocated_presudo
            self.presudo_labels0 = reallocated_presudo0
            
            self.data.X = reallocated_data
            self.data.y = reallocated_label
            self.Prototype = torch.stack(Prototype).cpu().numpy()
            self.GraphA, weight_raw = cal_weights_via_CAN(torch.tensor(self.data.X, dtype=torch.float32).to(self.device), self.num_neighbors)
            
            self.mean_sensitivity = calculate_prototype_sensitivity([X.cpu().numpy()], self.num_clusters)
            self.graph_sensitivity = calculate_graph_sensitivity([X.cpu().numpy()], device, self.num_neighbors)
            print('mean_sensitivity:', self.mean_sensitivity)
            print('graph_sensitivity:', self.graph_sensitivity)
        else:
            self.encoder.train(X.T, self.GlobalS, num_epochs, lr, momentum, weight_decay, init=self.init_)
            Embedding, self.GraphA = self.encoder(X.T)
            
            self.gmm = GaussianMixture(n_components=self.num_clusters, covariance_type='full', n_init=20, init_params='random', random_state=42).fit(Embedding.detach().cpu().numpy())
            self.presudo_labels = torch.tensor(self.gmm.predict(Embedding.detach().cpu().numpy())).to(self.device)
            self.Prototype = self.gmm.means_
            
            """
            self.presudo_labels, S, evs, cs = CLR(self.GraphA, self.num_clusters, 0, 0)
            self.presudo_labels = torch.tensor(self.presudo_labels).to(device)
            
            """
            
            reallocated_data = torch.zeros_like(X).to(self.device)
            reallocated_label = torch.zeros_like(y).to(self.device)
            reallocated_presudo = torch.zeros_like(self.presudo_labels).to(self.device)
            
            Prototype = []
            
            cnt_pos = 0
            for i in np.random.choice(range(self.num_clusters), self.num_clusters, replace=False):
                for j in range(torch.sum(self.data.y == i)):
                    reallocated_data[:, cnt_pos] = X[:, self.data.y == i][:, j]
                    reallocated_label[cnt_pos] = y[self.data.y == i][j]
                    reallocated_presudo[cnt_pos] = self.presudo_labels[self.data.y == i][j]
                    cnt_pos += 1
                Prototype.append(torch.mean(X[:, self.data.y == i], dim=1))
            
            self.presudo_labels = reallocated_presudo
            self.data.X = reallocated_data
            self.data.y = reallocated_label
            self.Prototype = torch.stack(Prototype).cpu().numpy()
            
            Embedding, self.GraphA = self.encoder(self.data.X.T)
        
        return self.presudo_labels, self.GraphA
    
    def Cal_CAN(self):
        X = self.data.X
        if self.init_:
            A, weight_raw = cal_weights_via_CAN(torch.tensor(X, dtype=torch.float32).to(self.device), self.num_neighbors)
        else:
            Embedding, A = self.encoder(X.T)
        return A
    
    def upload(self):
        return self.GraphA, self.Prototype, self.presudo_labels, self.data.y, self.mean_sensitivity, self.graph_sensitivity


def row_normalize(C):
    # Calculate the L2 norm of each row
    row_norms = torch.norm(C, p=2, dim=1, keepdim=True)
    
    # Avoid division by zero
    row_norms[row_norms == 0] = 1

    # Row-wise normalization
    C_normalized = C / row_norms
    return C_normalized


class Server:
    def __init__(self, num_clients, num_clusters, num_neighbors, num_sharing, epsilon_total, device='cuda:1'):
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.num_neighbors = num_neighbors
        self.device = device
        self.num_sharing = num_sharing
        self.epsilon_total = epsilon_total
        
    def pull(self, clients):
        self.GraphA = []
        self.Means = []
        self.presudo_labels = []
        self.presudo_labels0 = []
        self.private_clusters = []
        self.private_samples_num = []
        self.ground_truth_labels = []
        self.ground_truth_labels_all = []
        self.sensitivity_graph = []
        self.sensitivity_prototype = []
        
        epsilon_client = self.epsilon_total / self.num_clients  # Allocate privacy budget
        for client in clients:
            grapha, prototype, presudo_labels, ground_truth, mean_sensitivity, grapha_sensitivity = client.upload()
            self.sensitivity_graph.append(grapha_sensitivity)
            self.sensitivity_prototype.append(mean_sensitivity)
            
            class_counter = np.unique(ground_truth.cpu().detach().numpy(), return_counts=True)[1]
            
            epsilon_mean = epsilon_client * (mean_sensitivity / (mean_sensitivity + grapha_sensitivity))  # Allocate privacy budget
            epsilon_graph = epsilon_client * (grapha_sensitivity / (mean_sensitivity + grapha_sensitivity))  # Allocate privacy budget
            
            
            noisy_graph = add_noise_to_graph(grapha.cpu().detach().numpy(), grapha_sensitivity, epsilon_graph)
            self.GraphA.append(noisy_graph)
            
            epsilon_per_class = allocate_privacy_budget(class_counter, epsilon_mean)
            noisy_prototypes = laplace_noise_addition(prototype, mean_sensitivity, epsilon_per_class)
            self.Means.append(noisy_prototypes)
            
            #self.Means.append(prototype)
            print('total_epsilon: %f, client_epsilon: %f, epsilon_mean: %f, epsilon_graph: %f' % (self.epsilon_total, epsilon_client, epsilon_mean, epsilon_graph))
            
            print('class_counter:', class_counter)
            print('epsilon_per_class:', epsilon_per_class)
            print('mean_sensitivity:', mean_sensitivity)
            print('graph_sensitivity:', grapha_sensitivity)
            
            self.presudo_labels.append(presudo_labels.cpu().detach().numpy().astype(int))
            self.presudo_labels0.append(client.presudo_labels0.cpu().detach().numpy().astype(int))
            self.ground_truth_labels.append(ground_truth.cpu().detach().numpy().astype(int))
            self.private_clusters.append(np.unique(ground_truth.cpu().detach().numpy()))
            self.ground_truth_labels_all = np.concatenate((self.ground_truth_labels_all, ground_truth.cpu().detach().numpy()))
            self.private_samples_num = np.append(self.private_samples_num, len(ground_truth))

        self.private_samples_num = self.private_samples_num.astype(int)

    def Aggregate(self, Clients_list, k=3):

        # 1. Calculate the total number of samples across all clients
        total_samples = sum([client.data.X.shape[1] for client in Clients_list])
        global_graph = np.zeros((total_samples, total_samples))

        # 2. Record the starting index of each client in the global graph
        sample_idx = 0
        client_start_indices = []

        for i, client in enumerate(Clients_list):
            client_start_indices.append(sample_idx)
            num_samples = client.data.X.shape[1]
            start_idx = sample_idx
            end_idx = sample_idx + num_samples

            sample_idx += num_samples

        self.Matches = self.match_clusters_greedy(self.Means)
        
        i_rec_pos = 0

        # For each pair of clients
        for i in range(self.num_clients):
            j_rec_pos = 0
            for j in range(self.num_clients):
                pseudo_labels_i = self.presudo_labels0[i]
                pseudo_labels_j = self.presudo_labels0[j]

                # Ensure pseudo_labels_i and pseudo_labels_j are of integer type
                pseudo_labels_i = np.asarray(pseudo_labels_i, dtype=int)
                pseudo_labels_j = np.asarray(pseudo_labels_j, dtype=int)
                
                # Perform matching directly using integer indices
                match_matrix = np.array([
                    self.Matches[i][label_i] == self.Matches[j][label_j] 
                    for label_i in pseudo_labels_i for label_j in pseudo_labels_j
                ]).reshape(len(pseudo_labels_i), len(pseudo_labels_j))

                # Assign the matching result to global_graph
                global_graph[i_rec_pos: i_rec_pos + len(pseudo_labels_i), j_rec_pos: j_rec_pos + len(pseudo_labels_j)] = match_matrix.astype(float) * 0.03
                # global_graph[i * self.private_samples_num[i]: (i + 1) * self.private_samples_num[i],
                #             j * self.private_samples_num[j]: (j + 1) * self.private_samples_num[j]] = match_matrix.astype(float) * 0.03
                j_rec_pos += len(pseudo_labels_j)
            i_rec_pos += len(pseudo_labels_i)


        sample_idx = 0
        for i, client in enumerate(Clients_list):
            num_samples = client.data.X.shape[1]
            start_idx = sample_idx
            end_idx = sample_idx + num_samples
            
            # Fill the diagonal blocks (intra-client similarity)
            global_graph[start_idx:end_idx, start_idx:end_idx] = self.GraphA[i]
        
            sample_idx += num_samples
        
        # 8. Convert the global similarity matrix to a PyTorch tensor and move it to the specified device
        self.GlobalGraphA = torch.tensor(global_graph, dtype=torch.float32).to(self.device)

        return self.GlobalGraphA

    def match_clusters(self, Means):
        """
        Find the most similar cluster in client 1 (`Means[0]`) for each cluster of every other client 
        and return the matching results. Each client's clusters are matched against client 1's clusters 
        to avoid duplicate matching.
        
        Parameters:
        - Means: List of numpy arrays, each representing the cluster centroids for each client.
        
        Returns:
        - match_dict: A dictionary of matching results, where the key is the client and the value is the match with clusters in client 1.
        """
        from scipy.spatial.distance import euclidean
        from scipy.spatial.distance import cityblock
        
        match_dict = {0:{q: q for q in range(len(Means[0]))}}
        
        # Clusters of client 1
        client1_clusters = Means[0]
        
        # Match for each client
        for j in range(1, self.num_clients):
            client_j_clusters = Means[j]
            
            # Calculate the cosine similarity between each cluster of client j and each cluster of client 1
            sim_matrix = np.zeros((len(client_j_clusters), len(client1_clusters)))
            for q in range(len(client_j_clusters)):
                for p in range(len(client1_clusters)):
                    #sim_matrix[q, p] = euclidean(client_j_clusters[q].reshape(1, -1), client1_clusters[p].reshape(1, -1))
                    #sim_matrix[q, p] = cityblock(client_j_clusters[q].reshape(1, -1), client1_clusters[p].reshape(1, -1))
                    sim_matrix[q, p] = -cosine_similarity(client_j_clusters[p].reshape(1, -1), client1_clusters[q].reshape(1, -1))
            
            # Use the Hungarian algorithm for optimal matching
            row_ind, col_ind = linear_sum_assignment(sim_matrix)  # The negative sign is for maximizing similarity
            
            # Store the matching result in the dictionary
            #match_dict[j] =  {q : q for q in range(len(client_j_clusters))}
            match_dict[j] =  col_ind
            print("Match for client", j, ":", col_ind)
        
        return match_dict

    def match_clusters_neighbor(self, Means, k=3):
        """
        Find the k-nearest neighbors for each cluster center and perform matching based on the count of 
        mutual neighbors, completing the match between clusters of each client in Means and those of client 1.
        
        Parameters:
        - Means: List of numpy arrays, each representing the cluster centroids for each client.
        - k: int, Number of nearest neighbors for each cluster center, default is 3.
        
        Returns:
        - match_dict: A dictionary of matching results, where the key is the client and the value is the match with clusters in client 1.
        """
        match_dict = {0: {q: q for q in range(len(Means[0]))}}  # Client 1's clusters are matched with themselves
        
        # Clusters of client 1
        client1_clusters = Means[0]
        
        for client_idx in range(1, self.num_clients):  # use client_idx instead of outer j
            client_j_clusters = Means[client_idx]
            
            # Step 1: Calculate the k-nearest neighbors for each cluster center of client j
            nbrs_client_j = NearestNeighbors(n_neighbors=k).fit(client_j_clusters)
            distances_j, indices_j = nbrs_client_j.kneighbors(client1_clusters)  # k-nearest neighbors of client1_clusters with respect to client_j_clusters
            
            nbrs_client_1 = NearestNeighbors(n_neighbors=k).fit(client1_clusters)
            distances_1, indices_1 = nbrs_client_1.kneighbors(client_j_clusters)  # k-nearest neighbors of client_j_clusters with respect to client1_clusters
            
            # Step 2: Count the number of mutual k-nearest neighbors
            mutual_knn_count = np.zeros((len(client_j_clusters), len(client1_clusters)))
            for i in range(len(client_j_clusters)):
                for p in range(len(client1_clusters)):  # use p instead of inner j
                    if p in indices_1[i] and i in indices_j[p]:
                        mutual_knn_count[i, p] = sum([1 for idx in indices_1[i] if idx in indices_j[p]])
            
            # Step 3: Assign matching results
            matched_indices = [-1] * len(client_j_clusters)
            used_indices = set()
            for _ in range(min(len(client_j_clusters), len(client1_clusters))):
                max_mutual = -1
                best_match_i = -1
                best_match_p = -1
                for i in range(len(client_j_clusters)):
                    if matched_indices[i] != -1:
                        continue
                    for p in range(len(client1_clusters)):
                        if p not in used_indices and mutual_knn_count[i, p] > max_mutual:
                            max_mutual = mutual_knn_count[i, p]
                            best_match_i = i
                            best_match_p = p
                if best_match_i != -1 and best_match_p != -1:
                    matched_indices[best_match_i] = best_match_p
                    used_indices.add(best_match_p)
            
            match_dict[client_idx] = {i: matched_indices[i] for i in range(len(matched_indices)) if matched_indices[i] != -1}
            print("Match for client", client_idx, ":", match_dict[client_idx])
        
        return match_dict

    
    def match_clusters_greedy(self, Means):
        """
        Use a greedy strategy to match cluster centers between client1_clusters and client_j_clusters.
        
        Parameters:
        - Means: List of numpy arrays, each representing the cluster centroids for each client.
        
        Returns:
        - match_dict: A dictionary of matching results, where the key is the client and the value is the match with clusters in client 1.
        """
        from scipy.spatial.distance import cdist

        match_dict = {0: {q: q for q in range(len(Means[0]))}}  # Client 1's clusters are matched with themselves
        
        # Clusters of client 1
        client1_clusters = Means[0]
        
        # Match for each client
        for j in range(1, self.num_clients):
            client_j_clusters = Means[j]
            
            # Step 1: Calculate the distance matrix
            distance_matrix = cdist(client_j_clusters, client1_clusters, metric='euclidean')  # Euclidean distance
            
            # Step 2: Greedy assignment
            matched_indices = [-1] * len(client_j_clusters)  # Initialize matching results
            used_client1_indices = set()  # Keep track of matched cluster indices in client1_clusters
            
            while True:
                min_distance = float('inf')
                best_match_i = -1
                best_match_p = -1
                
                # Iterate through unmatched clusters to find the pair with the minimum distance
                for i in range(len(client_j_clusters)):
                    if matched_indices[i] != -1:  # Skip already matched clusters
                        continue
                    for p in range(len(client1_clusters)):
                        if p in used_client1_indices:  # Skip already matched clusters
                            continue
                        if distance_matrix[i, p] < min_distance:
                            min_distance = distance_matrix[i, p]
                            best_match_i = i
                            best_match_p = p
                
                # If no new matching pair is found, break the loop
                if best_match_i == -1 or best_match_p == -1:
                    break
                
                # Update the matching results
                matched_indices[best_match_i] = best_match_p
                used_client1_indices.add(best_match_p)
            
            # Step 3: Store the matching results in the dictionary
            match_dict[j] = {i: matched_indices[i] for i in range(len(matched_indices)) if matched_indices[i] != -1}
            print("Match for client", j, ":", match_dict[j])
        
        return match_dict

    def train(self):
        y, S, evs, cs = CLR(self.GlobalGraphA, self.num_clusters, 0, 0, device=self.device)
        self.GlobalS = S
        return y, S
    
    def Synchronize(self, clients):
        num_samples = self.GlobalS.shape[0]
        for i, client in enumerate(clients):
            
            # The j-th original sample corresponds to the Sample_map_re[j]-th sample in the sorted samples
            Sample_map_re = np.zeros(num_samples, dtype=np.int32)
            for j in range(num_samples):
                Sample_map_re[j] = np.where(self.Sample_Map[i] == j)[0][0]
            
            GlobalS = torch.zeros_like(self.GlobalS).to(self.device)
            
            GlobalS = self.GlobalS[:, Sample_map_re]
            GlobalS = GlobalS[Sample_map_re, :]
            
            client.init_ = False
            client.GlobalS = GlobalS.clone()


class Dataset:
    def __init__(self, X, y, SX, SY, A=None):
        self.X = X
        self.y = y
        self.A = A

        self.SX = SX
        self.SY = SY

from sklearn.metrics import confusion_matrix

def bestMap(L1, L2):
    # Ensure label vectors are 1D
    L1 = np.array(L1).flatten()
    L2 = np.array(L2).flatten()

    if L1.shape != L2.shape:
        raise ValueError("Size of L1 must equal size of L2")

    # Get unique labels and their counts
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)

    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass), dtype=int)
    # Build the cost matrix
    for i in range(nClass1):
        for j in range(nClass2):
            G[i, j] = np.sum(np.bitwise_and((L1 == Label1[i]), (L2 == Label2[j])))
    print(G)
    # Use the Hungarian algorithm to find the best assignment
    row_ind, col_ind = linear_sum_assignment(-G)
    # Remap L2 labels to match L1
    newL2 = np.zeros(L2.shape, dtype=int)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[col_ind[i]]

    return newL2

import numpy as np

def sampling(X, y, num_classes=10, num_clients=5, num_shards=1000, sharing_samples_num=100, heterogeneity=0.5):
    arg_sort = np.argsort(y)
    y_sorted = y[arg_sort]
    
    class_st_pos_dict = np.zeros(num_classes).astype(int)
    class_sum = np.zeros(num_classes).astype(int)

    select_class_num = np.ones((num_clients, num_classes)).astype(int)
    selected_main_class = np.random.choice(range(num_classes), num_clients)
    for i in range(num_clients):
        for j in range(num_classes):
            if j == selected_main_class[i]:
                select_class_num[i, j] = num_shards + heterogeneity * num_shards * (num_classes - 1)
            else:
                select_class_num[i, j] = (1 - heterogeneity) * num_shards
    
    select_class_num = np.array(select_class_num).astype(int)
    
    for i in range(1, num_classes):
        class_st_pos_dict[i] = np.sum(y_sorted == i-1) + class_st_pos_dict[i-1]
    for i in range(num_classes):
        class_sum[i] = np.sum(y_sorted == i)
    
    # Step 1: First, select the shared samples
    # Calculate the number of shared samples for each class
    class_sharing_samples = np.zeros(num_classes).astype(int)
    for i in range(num_classes):
        class_sharing_samples[i] = sharing_samples_num // num_classes
    # If there are remaining shared samples, distribute them evenly to each class
    remaining_sharing_samples = sharing_samples_num - np.sum(class_sharing_samples)
    for i in range(remaining_sharing_samples):
        class_sharing_samples[i] += 1
    
    shared_samples = []
    for i in range(num_classes):
        class_start = class_st_pos_dict[i]
        class_end = class_st_pos_dict[i+1] if i < num_classes - 1 else len(X)
        # Uniformly select `class_sharing_samples[i]` shared samples from each class
        shared_samples += list(np.random.choice(range(class_start, class_end), 
                                                class_sharing_samples[i], replace=False))
    
    # Step 2: Perform client data allocation, excluding the already selected shared samples
    remaining_samples = list(set(range(len(X))) - set(shared_samples))
    id_clients = {}
    
    for i in range(num_clients):
        id_clients[i] = []
        for j in range(num_classes):
            # Calculate the remaining samples for each class
            if j == num_classes - 1:
                selected_id = np.random.choice(range(class_st_pos_dict[j], len(X)), 
                                               min(select_class_num[i, j], class_sum[j]), replace=False)
            else:
                selected_id = np.random.choice(range(class_st_pos_dict[j], class_st_pos_dict[j+1]), 
                                               min(select_class_num[i, j], class_sum[j]), replace=False)
            # Exclude the already selected shared samples
            selected_id = list(set(selected_id) - set(shared_samples))
            id_clients[i] += list(arg_sort[selected_id])
    
    return shared_samples, id_clients
