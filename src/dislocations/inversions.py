from typing import Union
import numpy as np
import numba
import pygmo as pg
from scipy.linalg import block_diag


class InversionProblem:
    def __init__(self, design_matrix: np.ndarray, results_array: np.ndarray, weights_array: np.ndarray):
        assert all([isinstance(a, np.ndarray) for a in [design_matrix, results_array, weights_array]])
        assert design_matrix.ndim == 2
        assert all([a.ndim == 1 for a in [results_array, weights_array]])
        assert results_array.shape == weights_array.shape
        assert design_matrix.shape[0] == results_array.size
        self.design_matrix = design_matrix
        self.results_array = results_array
        self.weights_array = weights_array
        self.num_residuals = len(results_array)

    def fitness(self, x: np.ndarray):
        calc_disps = np.matmul(self.design_matrix, x)
        weighted_residuals = (calc_disps - self.results_array) * self.weights_array
        weighted_rms = np.linalg.norm(weighted_residuals)
        return np.array([weighted_rms])

    def get_bounds(self):
        num_params = self.design_matrix.shape[1]
        return [0.] * num_params, [10.] * num_params

    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)


class HawkeBayInversion:
    def __init__(self, gf_array: np.ndarray, laplacian: np.ndarray, edge_array: np.ndarray,
                 disp_weight: float, laplacian_weight: float, edge_weight: float,
                 max_slip: float = 8.):
        assert all([isinstance(a, np.ndarray) for a in [gf_array, laplacian, edge_array]])
        assert laplacian.ndim == 2
        assert all([a.ndim == 1 for a in [gf_array, edge_array]])
        assert gf_array.shape == edge_array.shape
        assert laplacian.shape[0] == gf_array.size

        self.gf_array = gf_array
        self.laplacian = laplacian
        self.edge_array = edge_array
        self.disp_weight = disp_weight
        self.laplacian_weight = laplacian_weight
        self.edge_weight = edge_weight
        self.max_slip = max_slip

    def get_bounds(self):
        num_params = self.gf_array.size
        return [0.] * num_params, [float(self.max_slip)] * num_params

    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)

    def fitness(self, x: np.ndarray):
        disp_contribution = np.dot(self.gf_array, x) * self.disp_weight
        laplacian_contribution = np.linalg.norm(np.matmul(self.laplacian, x)) * self.laplacian_weight
        edge_contribution = np.linalg.norm(np.dot(self.edge_array, x)) * self.edge_weight

        combined_fitness = disp_contribution + laplacian_contribution + edge_contribution
        return np.array([combined_fitness])


class EdgecumbeInversion:
    def __init__(self, edge_gf_array: np.ndarray, edge_laplacian: np.ndarray, edge_edge: np.ndarray,
                 awa_gf_array: np.ndarray, awa_laplacian: np.ndarray, awa_edge: np.ndarray,
                 edge_disp_weight: float, edge_laplacian_weight: float, edge_edge_weight: float,
                 awa_disp_weight: float, awa_laplacian_weight: float, awa_edge_weight: float,
                 displacements: np.ndarray, max_slip: float = 4.):
        assert all([isinstance(a, np.ndarray) for a in [edge_gf_array, edge_laplacian, edge_edge,
                                                        awa_gf_array, awa_laplacian, awa_edge]])
        assert all([arr.ndim == 2 for arr in [edge_gf_array, edge_laplacian,
                                              awa_gf_array, awa_laplacian]])
        assert all([a.ndim == 1 for a in [edge_edge, awa_edge]])


        self.edge_gfs, self.awa_gfs = edge_gf_array, awa_gf_array
        self.combined_gfs = np.hstack((edge_gf_array, awa_gf_array))
        self.edge_laplacian, self.awa_laplacian = edge_laplacian, awa_laplacian
        self.combined_laplacian = block_diag(edge_laplacian, awa_laplacian)
        self.edge_edge, self.awa_edge = edge_edge, awa_edge
        self.combined_edge = np.hstack((self.edge_edge, self.awa_edge))

        self.edge_disp_weight, self.awa_disp_weight = edge_disp_weight, awa_disp_weight
        self.combined_disp_weight = max([self.edge_disp_weight, self.awa_disp_weight])
        self.edge_laplacian_weight, self.awa_laplacian_weight = edge_laplacian_weight, awa_laplacian_weight
        self.combined_laplacian_weight = max([self.edge_laplacian_weight, self.awa_laplacian_weight])

        self.edge_edge_weight, self.awa_edge_weight = edge_edge_weight, awa_edge_weight

        self.combined_edge_weight = max([self.edge_edge_weight, self.awa_edge_weight])
        self.observed_disps = displacements
        self.max_slip = max_slip

        self.num_edge = len(self.edge_edge)
        self.num_awa = len(self.awa_edge)

    def get_bounds(self):
        num_params = self.num_edge + self.num_awa
        return [0.] * num_params, [float(self.max_slip)] * num_params

    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)

    def fitness(self, x: np.ndarray):
        disp_contribution = self.combined_disp_weight * np.linalg.norm(np.matmul(self.combined_gfs, x) - self.observed_disps)

        laplacian_contribution = np.linalg.norm(np.matmul(self.combined_laplacian, x)) * self.combined_laplacian_weight
        edge_contribution = np.linalg.norm(np.dot(self.combined_edge, x)) * self.combined_edge_weight

        combined_fitness = disp_contribution + laplacian_contribution + edge_contribution
        return np.array([combined_fitness])








