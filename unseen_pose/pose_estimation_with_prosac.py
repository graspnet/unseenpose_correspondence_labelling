import random
from abc import ABC, abstractmethod

import numpy as np
from scipy.stats.distributions import chi2
from scipy import stats
from .ransac_with_icp import solve_transformation_matrix
import matplotlib.pyplot as plt


class Model(ABC):
    @abstractmethod
    def fit(self, pts):
        ...

    @abstractmethod
    def error(self, data):
        ...

    @abstractmethod
    def predict(self, data):
        ...

    @staticmethod
    @abstractmethod
    def get_complexity():
        ...


class TransformationModel(Model):

    def __init__(self):
        self.R = np.identity(3)
        self.t = np.zeros(3)
        self.T = np.identity(4)

    def fit(self, data):
        A = data[:, :3]
        B = data[:, 3:]
        self.T = solve_transformation_matrix(A, B)
        self.R = self.T[:3, :3]
        self.t = self.T[:3, -1]

    def predict(self, X):
        assert len(X.shape) == 2 and X.shape[1] == 3
        Y = np.matmul(self.T, np.column_stack((X, np.ones((X.shape[0], 1)))).T).T
        Y = np.delete(Y, 3, axis=1)
        return Y

    def error(self, data):
        prediction = self.predict(data[:, :3])
        true_value = data[:, 3:]
        return np.abs(true_value - prediction).sum(axis = 1)

    @staticmethod
    def get_complexity():
        return 6


class LinearModel(Model):

    def __init__(self):
        self.m = 0
        self.b = 0

    def fit(self, pts):
        xs = pts[:, 0]
        ys = pts[:, 1]
        self.m, self.b = stats.linregress(xs, ys)[:2]

    def predict(self, x):
        return self.m * x + self.b

    def error(self, data):
        prediction = self.predict(data[:, 0])
        true_value = data[:, 1]
        return np.sqrt(np.square(true_value - prediction))

    @staticmethod
    def get_complexity():
        return 2


def ransac(data, model_type, tolerance, prob_inlier, p=0.99):
    """
    Random sample consensus (RANSAC)
    :param data: Data to fit
    :param model_type: Model subclass
    :param tolerance: Tolerance on the error to consider a point inlier to a model
    :param prob_inlier: Probability that a point is an inlier (inliers / (inliers + outliers))
    :param p: Desired probability that there is at least one good sample
    :return: A model of type model_type, fitted to the inliers
    """
    m = model_type.get_complexity()
    best_num_inliers = 0
    n = data.shape[0]
    max_times = int(np.ceil(np.log(1 - p) / np.log(1 - prob_inlier ** m)))
    satisfactory_inlier_ratio = prob_inlier * n

    inliers = []
    for _ in range(max_times):
        pts = data[random.sample(range(n), m)]
        model = model_type()
        model.fit(pts)
        error = model.error(data)
        num_inliers = (error < tolerance).sum()
        if num_inliers / n > satisfactory_inlier_ratio:
            inliers = data[error < tolerance]
            break
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            inliers = data[error < tolerance]

    model = model_type()
    model.fit(inliers)

    return model


def prosac(data, quality, model_type, tolerance, beta, eta0, psi,
           max_outlier_proportion, p_good_sample, max_number_of_draws,
           enable_n_star_optimization=True):
    """
    Progressive random sampling algorithm (PROSAC)
    Adapted from: http://devernay.free.fr/vision/src/prosac.c
    :param data: Data to fit
    :param quality: Point quality
    :param model_type: Model subclass
    :param tolerance: Tolerance on the error to consider a point inlier to a model
    :param beta: Probability that a match is declared inlier by mistake, i.e. the ratio of the "inlier"
    :param eta0: Maximum probability that a solution with more than In_star inliers in Un_star exists and was not found after k samples (typically set to 5%, see Sec. 2.2 of [Chum-Matas-05]).
    :param psi: Probability that In_star out of n_star data points are by chance inliers to an arbitrary (typically set to 5%)
    :param max_outlier_proportion: Maximum allowed outliers proportion in the input data, used to compute T_N (can be as high as 0.95)
    :param p_good_sample: Probability that at least one of the random samples picked up by RANSAC is free of outliers
    :param max_number_of_draws: Max number of draws
    :param enable_n_star_optimization: Enable early stopping if the probability of finding a better match fall below eta0
    :return: A model of type model_type, fitted to the inliers
    """
    indexes = np.argsort(quality)
    data = data[indexes[::-1]]

    num_points = data.shape[0]
    num_points_to_sample = model_type.get_complexity()
    chi2_value = chi2.isf(2 * psi, 1)

    def niter_ransac(p, epsilon, s, n_max):
        """
        Compute the maximum number of iterations for RANSAC
        :param p: Probability that at least one of the random samples picked up by RANSAC is free of outliers
        :param epsilon: Proportion of outliers
        :param s: Sample size
        :param n_max: Upper bound on the number of iterations (-1 means INT_MAX)
        :return: maximum number of iterations for RANSAC
        """
        if n_max == -1:
            n_max = np.iinfo(np.int32).max
        if not (n_max >= 1):
            raise ValueError('n_max must be positive')
        if epsilon <= 0:
            return 1
        logarg = - np.exp(s * np.log(1 - epsilon))
        logval = np.log(1 + logarg)
        n = np.log(1 - p) / (logval + 10 ** (-4))
        if logval < 0 and n < n_max:
            return np.ceil(n)
        return n_max

    def i_min(m, n, beta):
        """
        Non-randomness, prevent from choosing a model supported by outliers
        :param m: Model complexity
        :param n: Number of considered points
        :param beta: Beta parameter
        :return: Minimum number of inlier to avoid model only supported by outliers
        """
        mu = n * beta
        sigma = np.sqrt(n * beta * (1 - beta))
        return np.ceil(m + mu + sigma * np.sqrt(chi2_value))

    N = num_points
    m = num_points_to_sample
    T_N = niter_ransac(p_good_sample, max_outlier_proportion, num_points_to_sample, -1)
    I_N_min = (1 - max_outlier_proportion) * N

    if m >= N:
        return model_type().T

    n_star = N
    I_n_star = 0
    I_N_best = 0
    t = 0
    n = m
    T_n = T_N

    for i in range(m):
        T_n = T_n * (n - i) / (N - i)

    T_n_prime = 1
    k_n_star = T_N

    while ((I_N_best < I_N_min) or t <= k_n_star) and t < T_N and t <= max_number_of_draws:
        t = t + 1

        if (t > T_n_prime) and (n < n_star-1):
            T_nplus1 = (T_n * (n + 1)) / (n + 1 - m)
            n = n + 1
            T_n_prime = T_n_prime + np.ceil(T_nplus1 - T_n)
            T_n = T_nplus1

        if t > T_n_prime:
            pts_idx = np.random.choice(n, m, replace=False)
        else:
            pts_idx = np.append(np.random.choice(n - 1, m - 1, replace=False), n)

        sample = data[pts_idx]

        # 3. Model parameter estimation
        model = model_type()
        model.fit(sample)

        # 4. Model verification
        error = model.error(data)
        is_inlier = (error < tolerance)
        I_N = is_inlier.sum()

        if I_N >= I_N_best:
            I_N_best = I_N
            n_best = N
            I_n_best = I_N
            best_model = model

            if enable_n_star_optimization:
                epsilon_n_best = I_n_best / n_best
                I_n_test = I_N
                for n_test in range(N, m, -1):
                    if not (n_test >= I_n_test):
                        raise RuntimeError('Loop invariant broken: n_test >= I_n_test')
                    if ((I_n_test * n_best > I_n_best * n_test) and (I_n_test > epsilon_n_best * n_test + np.sqrt(
                            n_test * epsilon_n_best * (1 - epsilon_n_best) * chi2_value))):
                        if I_n_test < i_min(m, n_test, beta):
                            break
                        n_best = n_test
                        I_n_best = I_n_test
                        epsilon_n_best = I_n_best / n_best
                    I_n_test = I_n_test - is_inlier[n_test - 1]

            if I_n_best * n_star > I_n_star * n_best:
                if not (n_best >= I_n_best):
                    raise RuntimeError('Assertion not respected: n_best >= I_n_best')
                n_star = n_best
                I_n_star = I_n_best
                k_n_star = niter_ransac(1 - eta0, 1 - I_n_star / n_star, m, T_N)

    return best_model


# GENERATE DATA FOR TEST PURPOSE
def generate_data(n, outliers, R, t):
    data = np.zeros((n + outliers, 6))

    X = np.random.rand(n, 3)
    X[:, 2] = 0.0
    Y = (R @ X.T + t.reshape(-1,1)).T
    data[:n, :3] = X
    data[:n, 3:] = Y

    X_out = np.random.rand(outliers, 3) - 0.5
    Y_out = np.random.rand(outliers, 3) - 0.5

    data[n:, :3] = X_out
    data[n:, 3:] = Y_out

    quality = np.append(1 / abs(np.random.randn(n)), 1 / abs(np.random.randn(outliers) * 2 + 3))
    return data, quality


if __name__ == '__main__':
    m = -4
    b = 10

    R = np.array([[0.28478417, 0.09621854, 0.95375048],
                  [0.73874211, 0.6120067, -0.28232588],
                  [-0.61086666, 0.78497758, 0.10320913]])

    t = np.array([0.0, 0.0, 1.5])

    inliers = 100
    outliers = 100
    data, quality = generate_data(inliers, outliers, R, t)

    X, Y = data[:, :3], data[:, 3:]

    import open3d as o3d
    p1 = o3d.geometry.PointCloud()
    p1.points = o3d.utility.Vector3dVector(X)
    p2 = o3d.geometry.PointCloud()
    p2.points = o3d.utility.Vector3dVector(Y)
    p1.paint_uniform_color([0.2, 0.9, 0.1])
    p2.paint_uniform_color([0.5, 0.2, 0.1])

    o3d.visualization.draw_geometries([p1, p2])

    tolerance = 1

    # Base from data ranges and tolerance, a point has beta probability of being supported by an incorrect model
    # data_area = (x_max - x_min) * (y_max - y_min)
    # max_line_length = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
    # tolerance_area = max_line_length * tolerance

    # The value can be hard to estimate, it is better to underestimate the value
    prob_inlier = inliers / (inliers + outliers) * 0.9

    # Desired probability that there is at least one good sample
    p = 0.99

    # beta = (tolerance_area / data_area) * 1.1  # add 10% to be more pessimist
    model_prosac = prosac(data, quality, TransformationModel, tolerance, 0.2,
                          eta0=0.05, psi=0.05, max_outlier_proportion=(1 - prob_inlier),
                          p_good_sample=p, max_number_of_draws=60_000,
                          enable_n_star_optimization=True)

    model_ransac = ransac(data, TransformationModel, tolerance, prob_inlier=prob_inlier, p=p)

    print(model_ransac.T)
    print(model_prosac.T)

    p1.transform(model_prosac.T)
    o3d.visualization.draw_geometries([p1, p2])


def pose_estimation_point_clouds_with_prosac(pcd_l, pcd_r, matches):
    """
    TO DO
    """
    xyz_l = np.asarray(pcd_l.points)
    xyz_r = np.asarray(pcd_r.points)

    data = np.zeros((matches.shape[0], 6))
    data[:, :3] = xyz_l[matches[:, 0].astype(np.int32)]
    data[:, 3:] = xyz_r[matches[:, 1].astype(np.int32)]
    quality = matches[:, 3]
    model_prosac = prosac(data, quality, TransformationModel, 0.01, 0.2,
                          eta0=0.05, psi=0.05, max_outlier_proportion=(1 - 0.5),
                          p_good_sample=0.5, max_number_of_draws=60_000,
                          enable_n_star_optimization=False)
    return model_prosac.T
