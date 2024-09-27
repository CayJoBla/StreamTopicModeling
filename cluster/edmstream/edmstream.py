import math
import numpy as np
from typing import Optional

from .cluster_cell import ClusterCell
from .dptree import DPTree

class EDMStream:
    """My custom implementation of the EDMStream algorithm. 

    See the original technical paper for more details:
    "Clustering Stream Data by Exploring the Evolution of Density Mountain"
    (https://arxiv.org/pdf/1710.00867)

    
    Parameters:
        decay_factor (float): The fraction of density to retain per time step.
            In terms of the paper, this is the value of (a ^ lambda).
        alpha (float): The balance between inter-cell and intra-cell distances
            for dynamic distance thresholding in the DP-Tree.
        beta (float): The fraction of total density needed for an active cell.
        epsilon (float): The radius of the cluster-cell neighborhood. 
        stream_speed (int): The number of points per time step in the stream.
        num_initial_cells (int): The number of active clusters needed to 
            initialize the DP-Tree.
        term_threshold (float): The minimum term frequency for a term to be
            retained in the c-TF-IDF values of a cluster-cell.
    """
    def __init__(
        self,
        decay_factor: float = 0.98,
        alpha: Optional[float] = 0.5,   # TODO: Finalize decision on how to choose this
        beta: float = 2e-3,
        epsilon: float = 0.1,
        stream_speed: int = 128,
        num_initial_cells: int = 10,
        term_threshold: float = 0.1,
    ):
        # Initialize algorithm parameters
        self.timestamp = 0
        if not (0 < decay_factor < 1):  # Check decay factor input value
            raise ValueError("Decay factor must be between 0 and 1.")
        self.decay_factor = decay_factor    # a = decay_factor, lambda = 1
        self.decay_fn = lambda t: self.decay_factor ** t
        self.stream_speed = stream_speed
        self.radius = epsilon

        # Initialize outlier parameters
        max_total_density = self.stream_speed / (1-self.decay_factor)
        if not (1/max_total_density < beta < 1):    # Check beta input value
            raise ValueError("Beta value is outside the acceptable range.")
        self.density_threshold = beta * max_total_density
        self.T_del = math.log(1/self.density_threshold, self.decay_factor) \
                        / self.stream_speed
        self.t_last_outlier_check = self.timestamp
        self.outlier_cells = []     # Outlier reservoir

        self._n_samples_seen = 0
        self._t_per_sample = 1 / self.stream_speed
        self.term_threshold = term_threshold

        self.tree = DPTree(num_initial_cells)
        self.alpha = alpha

    def assign(self, x, document=None):
        """Assigns a new sample to a cluster-cell. First tries to assign to an 
        existing cluster-cell, but creates a new cluster-cell for an outlier
        point if necessary. 

        Args:
            x (np.ndarray): The new sample to assign to a cluster-cell.

        Returns:
            (int): The ID of the cluster-cell to which the sample was assigned.
        """
        # Combine the active and outlier cluster-cells
        all_cluster_cells = self.tree.cluster_cells + self.outlier_cells
        seeds = np.array([cell.seed for cell in all_cluster_cells])

        # Try to assign the point to an existing cluster-cell
        if len(all_cluster_cells) > 0:
            distances = np.linalg.norm(seeds - x, axis=-1)
            closest_cell_idx = np.argmin(distances)

            if distances[closest_cell_idx] <= self.radius:
                cluster_cell = all_cluster_cells[closest_cell_idx]
                if cluster_cell.id in self.tree.keys(): # Active cluster-cell 
                    self.tree.assign_point_to_cell(
                        cluster_cell.id, 
                        self.timestamp, 
                        distances[:len(self.tree)], 
                        document
                    )
                else:                                   # Outlier cluster-cell
                    cluster_cell.insert_one(self.timestamp, document)
                    density = cluster_cell.get_density(self.timestamp)
                    if density >= self.density_threshold:
                        self.tree.insert(cluster_cell, self.timestamp)
                        self.outlier_cells.remove(cluster_cell)

                return cluster_cell.id

        # Create a new outlier cluster-cell  
        cluster_cell = ClusterCell(
            seed = x,
            timestamp = self.timestamp,
            decay_fn = self.decay_fn,
            term_threshold = self.term_threshold,
            document = document
        )
        self.outlier_cells.append(cluster_cell)

        return cluster_cell.id

    def learn_one(self, x, document=None):
        """Processes a new sample from the stream."""
        if self.timestamp == 0:
            self.ndim = x.shape[0]
        elif x.shape[0] != self.ndim:   # Check input feature size
            raise ValueError(f"Input features ({x.shape[0]}) do not match the "\
                                f"expected size ({self.ndim}).")
        
        self.timestamp += self._t_per_sample    # Update the timestamp

        # Assign new point to a cluster-cell
        cluster_cell_id = self.assign(x, document)

        # Remove decayed inactive cells from the tree
        new_outliers = self.tree.pop_inactive(self.timestamp, 
                                                self.density_threshold)
        self.outlier_cells.extend(new_outliers)

        # Remove outliers that have not received points in a while
        if self.timestamp - self.t_last_outlier_check >= self.T_del:
            self.outlier_cells = [
                cell for cell in self.outlier_cells if 
                (self.timestamp - cell.t_insert) <= self.T_del
            ]
            self.t_last_outlier_check = self.timestamp

        # TODO: REMOVE LATER; Check that dependency order is maintained
        for i, cell in enumerate(self.tree.cluster_cells[:-1]):
            if not self.tree.initialized:
                break
            density = cell.get_density(self.timestamp)
            dep_density = self.tree[cell.dependency].get_density(self.timestamp)
            if cell.dependency not in self.tree.ids[i+1:]:
                print("Timestamp:\t", self.timestamp)
                print("Updated cell:\t", cluster_cell_id)
                print("Cluster-cell:\t", cell.id)
                print("Dependency:\t", cell.dependency)
                if density > dep_density:
                    raise ValueError("Cluster-cell has a higher density than its dependency.")
                else:
                    raise ValueError("Cluster-cell dependency is not in the tree.")
            else:
                if density > dep_density:
                    raise ValueError("Density order has not been maintained.")

    def _get_dynamic_tao(self):
        """This function chooses the distance threshold (tao) that minimizes
        a loss function that balances inter-cell and intra-cell distances.

        Note that the loss function here differs from the loss function in the 
        paper, which appears to be incorrect. This loss function minimizes the
        average intra-cell distance while maximizing the average inter-cell
        distance. The alpha parameter accounts for the user preference between
        these two objectives.
        """
        # Check cell count and address potential issues
        num_cells = len(self.tree._cluster_cells)
        if num_cells <= 1:  # If no cells, then tao doesn't matter``
            return 0        # If one cell, then cell is its own cluster

        # Sort dependent distances and drop the cell with no dependencies
        dependent_dists = np.sort([cell.dependent_dist for cell in 
                                    self.tree._cluster_cells.values()])[:-1]
        tao_vals = np.insert(dependent_dists, 0, 0)     # Add tao = 0
        cum_dist_sum = np.cumsum(tao_vals)
        mean_dist = np.mean(dependent_dists)
        total_sum = cum_dist_sum[-1]

        # Compute the loss function for each tao (custom loss)
        # print("\nVersion 1")
        # m = np.arange(1, num_cells)
        # n = num_cells - m
        # inter_cell_loss, intra_cell_loss = np.zeros((2,num_cells))
        # inter_cell_loss[:-1] = (cum_dist_sum[:-1] - cum_dist_sum[-1]) / n
        # intra_cell_loss[1:] = cum_dist_sum[1:] / m
        # print(inter_cell_loss)
        # print(intra_cell_loss)
        # loss = self.alpha * inter_cell_loss + (1-self.alpha) * intra_cell_loss
        # print("Loss:\t", loss)
        # print("Min tao:\t", tao_vals[np.argmin(loss)])

        # print("\nVersion 2")
        # total_sum = cum_dist_sum[-1]
        # print(cum_dist_sum)
        # inter_cell_loss, intra_cell_loss = np.zeros((2,num_cells))
        # inter_cell_loss[:-1] = (total_sum - cum_dist_sum[:-1]) / (n * mean_dist)
        # intra_cell_loss[1:] = (m * mean_dist) / cum_dist_sum[1:]
        # print(inter_cell_loss)
        # print(intra_cell_loss)
        # inter_cell_loss[:-1] *= self.alpha
        # intra_cell_loss[1:] *= (1-self.alpha)
        # loss2 = inter_cell_loss + intra_cell_loss
        # print("Loss2:\t", loss2)
        # print("Min tao2:\t", tao_vals[np.argmin(loss2)])

        print("\nVersion 3")
        m = np.arange(1, num_cells)
        n = num_cells - m
        inter_cell_loss, intra_cell_loss = np.zeros((2,num_cells))
        inter_cell_loss[:-1] = (n * mean_dist) / (total_sum - cum_dist_sum[:-1])
        intra_cell_loss[1:] = cum_dist_sum[1:] / (m * mean_dist)
        print(inter_cell_loss)
        print(intra_cell_loss)
        # inter_cell_loss[1:] *= self.alpha
        # intra_cell_loss[:-1] *= (1-self.alpha)
        # loss = inter_cell_loss + intra_cell_loss
        loss = self.alpha * inter_cell_loss + (1-self.alpha) * intra_cell_loss
        print("Loss:\t", loss)

        # Return the value of tao that results in the lowest loss
        return tao_vals[np.argmin(loss)]

    def predict(self, X, static_dist_threshold=None):
        # Check input shape
        X = np.atleast_2d(X)
        if not self.tree.initialized:
            raise ValueError("The DP-Tree has not been initialized, and so "
                                "predictions cannot be made. Consider training "
                                "the model with more data, or force "
                                "initialization with self.tree.initialize().")

        # Get the seeds of the active cluster-cells
        seeds = np.array([self.tree[cell_id].seed for cell_id in self.tree.ids])

        # Initialize cluster labels
        labels = [-1] * X.shape[0]
        if seeds.shape[0] == 0:     # No active cluster cells, all outliers
            return labels

        # Get the clusters from the DP-Tree
        distance_threshold = static_dist_threshold
        if distance_threshold is None:
            distance_threshold = self._get_dynamic_tao()
        clusters, cluster_labels = self.tree.query_subtrees(distance_threshold)

        # c_tf_idf = self._get_tf_idf(cluster_labels)
 
        # Label points with the cluster of the closest cell
        distances = np.linalg.norm(seeds[None,:,:] - X[:,None,:], axis=-1)
        closest_cell_indices = np.argmin(distances, axis=1)
        for i, close_idx in enumerate(closest_cell_indices):
            if distances[i][close_idx] <= self.radius:
                labels[i] = cluster_labels[self.tree.ids[close_idx]]
        
        return labels

    def _get_tf_idf(self, cluster_labels):
        unique_terms = set()

        # Get all unique words (TODO: Maybe track this globally instead?) 
        for cell_id in self.tree.keys():
            cell_tf = self.tree[cell_id].get_tf(self.timestamp)
            unique_terms.update(cell_tf.keys())

        # Build the term frequency matrix
        frequencies = np.zeros((len(self.tree), len(unique_terms)))
        unique_terms = list(unique_terms)
        for cell_id, cluster_id in cluster_labels.items():
            cell_tf = self.tree[cell_id].get_tf(self.timestamp)
            for i, term in enumerate(unique_terms):
                frequencies[cluster_id, i] += cell_tf.get(term, 0)

        # Compute the c-TF-IDF values
        A = np.mean(np.sum(frequencies, axis=1))
        fx = np.sum(frequencies, axis=0)
        idf = np.log(1 + A / fx)
        # TODO: Bertopic mentions taking the L1 norm of tf, what does that mean?
        return frequencies * idf