import numpy as np
from scipy.spatial.distance import cdist
import math

from dptree import DPTree


class EDMStream:
    """My custom implementation of the EDMStream algorithm. 

    See the original technical paper for more details:
    "Clustering Stream Data by Exploring the Evolution of Density Mountain"
    (https://arxiv.org/pdf/1710.00867)

    Parameters:
        decay_factor (float): The fraction of density to retain per time step.
            In terms of the paper, this is the value of (a ^ lambda).
        beta (float): The fraction of total density needed for an active cell.
        radius (float): The radius of the cluster-cell neighborhood. 
        batch_size (int): The number of points per batch. This is also 
            equivalent to the stream speed v in the paper.
        init_size (int): The minimum number of points to consider before 
            initializing the DP-Tree.
    """
    def __init__(
        self,
        decay_factor=.98,
        beta=4e-3,
        radius=0.1,
        batch_size=128,
        init_size=1024,
    ):
        super().__init__()
        self.timestamp = 0
        self.decay_factor = decay_factor

        # TODO: Set these values correctly
        max_total_density = self.batch_size / (1-self.decay_factor)
        if (1/max_total_density < beta < 1):
            raise ValueError("Beta value is outside the acceptable range.")
        self.density_threshold = beta * max_total_density
        self.radius = radius

        # TODO: Initialize needed attributes (Tdel, tao, etc.)
        self.dependency_threshold = None    # See paper
        self.T_del = math.log(1/self.density_threshold, self.decay_factor) \
                        / self.batch_size


        self._n_samples_seen = 0    # Deleted upon initialization
        self.batch_size = batch_size
        self.init_size = init_size
        self.initialized = False
        self.tree = None            # Fully initialized after init_size samples 

        # TODO: Implement an outlier reservoir instead of putting everything in the tree
        # Outlier reservoir
        self.outlier_seeds = []
        self.outlier_densities = []
        self.outlier_last_update_time = []


    def compute_best_tao(self):
        """Note that this fixes the loss function in the paper, which minimizes
        where it should maximize and vice versa. TODO: Check that this is true.
        """
        # Sort dependent distances and drop the cell with no dependencies
        distances = np.sort(self.cluster_cell_distances)[:-1]
        num_cells = len(distances)
        cum_dist_sum = np.insert(np.cum_sum(distances), 0, 0)
        total_dist_sum = cum_dist_sum[-1]
        mean_dist = total_dist_sum / num_cells

        # Compute the loss function for each tao
        m = np.arange(num_cells+1)
        n = num_cells - m
        inter_cell_loss = (n * mean_dist) / (total_dist_sum - cum_dist_sum)
        intra_cell_loss = cum_dist_sum / (m * mean_dist)

        # Find the tao that minimizes the loss function
        min_idx = np.argmin(alpha*inter_cell_loss + (1-alpha)*intra_cell_loss)
        return distances[min_idx-1] if min_idx > 0 else 0





        
        


    # TODO: If the storage structure of the cluster-cells changes, this needs to be updated
    @property
    def cluster_cell_seeds(self):
        # NOTE: This is a copy, you cannot use this to update the tree
        return self.tree.data['density']
    
    @property
    def cluster_cell_densities(self):
        # NOTE: This is a copy, you cannot use this to update the tree
        return self.tree.data['density']

    def get_update_filter(self, old_densities, distances, assignments):
        """Computes the complete update filter on the set of cluster-cells.
        This is done by computing both the density filter and the triangle-
        inequality filter, which have been adapted for the batchwise nature of 
        this implementation from theorems 1 and 2 in the paper.

        Args:
            old_densities (np.ndarray): The pre-update cell density values.
            distances (np.ndarray): The pairwise distance matrix between 
                                    cluster-cells and newly assigned points.
            point_assignments (np.ndarray): The cell assignments for each point.

        Returns:
            (np.ndarray): A boolean mask of the cluster-cells that need to be 
                          updated.
        """
        # Determine which cells have received new points
        new_densities = self.cluster_cell_densities
        is_updated = new_densities > old_densities
        print("Updated Cells Mask:", is_updated, sep="\n", end="\n\n")
        
        if not np.any(is_updated):  # No updated densities
            return np.zeros_like(is_updated, dtype=bool)

        # Compute the density filter
        is_less_old = np.less.outer(old_densities[is_updated], old_densities)
        is_less_new = np.less.outer(new_densities[is_updated], new_densities)
        density_causation = is_less_old != is_less_new
        density_filter = np.any(density_causation, axis=0)  # False = No Update
        print("Density Causation:", density_causation, sep="\n", end="\n\n")
        print("Density Filter:", density_filter, sep="\n", end="\n\n")

        if not np.any(density_filter):  # Density ordering unchanged
            return density_filter

        # Get the full causation mask from the updated and incidental cells
        is_update_cause = is_updated
        is_update_cause[is_updated] = np.any(density_causation, axis=1)
        print("Causation Mask:", is_update_cause, sep="\n", end="\n\n")

        # Compute the Tri-Inequality Filter for each incidental cell
        tri_ineq_filter = np.zeros(np.sum(density_filter), dtype=bool)
        dep_distances = np.array([cell.delta for cell in self.cluster_cells])
        for cell_idx in self.tree.ids[np.where(is_update_cause)[0]]:  
            print("------------------------------------------------")
            print("Cell ID:", cell_id, end="\n\n")

            point_mask = (assignments == cell_id)
            print("Assigned Point Mask:", point_mask, sep="\n", end="\n\n")

            differences = np.abs(np.subtract(
                distances[density_filter,   None,       point_mask],
                distances[None,             cell_id,    point_mask]
            ))
            print("Differences:", differences, sep="\n", end="\n\n")

            not_tri_ineq = (differences <= dep_distances[needs_update,None])
            print("Not Tri-Ineq:", not_tri_ineq, sep="\n", end="\n\n")

            tri_ineq_filter |= np.any(not_tri_ineq, axis=-1).ravel()
            print("Triangle Inequality Filter:", tri_ineq_filter, sep="\n", end="\n\n")

        return tri_ineq_filter

    def create_new_cells(self, X):
        """Adds new cluster-cells from a batch of outlier points.
        This implementation is greedy, choosing the point with the most 
        neighbors as the seed for the new cluster-cell at each iteration.

        Point assignments and new cluster-cells are returned.
        """
        # Which points are within the radius of each other
        close_points = (cdist(X,X) <= self.r)

        # Form cluster-cell groups
        seed_indices = []
        membership_masks = []
        while np.any(close_points):
            # Find the point with the most neighbors
            idx = np.argmax(close_points.sum(axis=1))
            members = close_points[idx]

            # Form neighbors into a new cluster-cell group
            seed_indices.append(idx)
            membership_masks.append(members)
            close_points[members], close_points[:,members] = False, False

        # Create the new cluster-cells
        seeds = X[seed_indices]
        membership_masks = np.array(membership_masks)
        densities = np.sum(membership_masks, axis=1)

        outlier_mask = densities < self.density_threshold
        active_mask = ~outlier_mask
        if np.any(active_mask):
            new_cell_ids = self.tree.add(
                seeds[active_mask], 
                densities[active_mask]
            )
        if np.any(outlier_mask):
            num_outliers = np.sum(outlier_mask)
            self.outlier_seeds.extend(seeds[outlier_mask])
            self.outlier_densities.extend(densities[outlier_mask])
            self.outlier_last_update_time.extend([self.timestamp]*num_outliers)





        return new_cell_ids[np.argmax(membership_masks, axis=0)]

    def assign(self, X):
        """Assigns new points to cluster-cells. First tries to assign points
        to existing cluster-cells, and then creates new cluster-cells for
        outlier points.

        Args:
            X (np.ndarray): The batch of new points to be assigned.

        Returns:
            (np.ndarray): The assignment labels of each point to a cluster-cell
            (np.ndarray): The distances between each point and the cluster-cells
            (list): The new cluster-cells created by the batch.
        """
        n_points, n_dims = X.shape
        
        # Assign points to existing cluster-cells
        assignments = np.full(num_points, -1, dtype=int)
        if self.tree.num_cluster_cells > 0:
            # Get the closest cluster cell for each point
            cell_point_dist = cdist(self.cluster_cell_seeds, X)
            closest_cell_idx = np.argmin(cell_point_dist, axis=0)

            # Increment the cluster cell densities where points are added
            inlier_mask = cell_point_dist[closest_cell_idx] <= self.r
            close_enough_idx = closest_cell_idx[inlier_mask]
            cell_mask = np.equal.outer(np.arange(num_cells), close_enough_idx)
            self.tree.densities += np.sum(cell_mask, axis=1)

            assignments[inlier_mask] = self.tree.ids[close_enough_idx]
            outlier_mask = ~inlier_mask
        else:
            distances = None
            outlier_mask = np.ones(num_points, dtype=bool)
        
        # Assign points to outlier cluster-cells
        if np.any(outlier_mask):
            new_cell_ids = self.create_new_cells(X[outlier_mask])
            assignments[outlier_mask] = new_cell_ids

        return assignments, distances

    def learn_batch(self, X):
        """Processes a batch of new points in the stream."""
        # Checks on the input data
        if X.ndim != 2:
            raise ValueError("Input data must be 2-dimensional")
        if X.shape != (self.batch_size, self.tree.ndim):
            raise ValueError("Incorrect input shape. Expected a shape of "
                             f"{self.batch_size}, but got {X.shape}.")

        # Compute timestep updates
        if self.initialized:
            self.timestamp += 1     # TODO: Determine the proper timestamp handling
            self.tree.densities *= self.decay_factor
        
        # Check outlier cells for memory space recycling
        pre_update_densities = self.cluster_cell_densities
        prune_mask = pre_update_densities < self.prune_threshold
        if np.any(prune_mask):
            self.tree.remove_by_idx(np.nonzero(prune_mask)[0])
        pre_update_densities = pre_update_densities[~prune_mask]

        # Assign points to cluster-cells and create new cluster-cells
        assignments, distances = self.assign(X)
        
        # Determine which cluster-cells need to be updated
        if self.initialized:
            # TODO: Remove this assertion after testing
            assert self.tree.num_cluster_cells > 0, "Initialized, but no cluster-cells exist..."
            update_filter = self.get_update_filter(
                pre_update_densities, distances, assignments
            )
        else:
            self._n_samples_seen += self.batch_size
            if self._n_samples_seen >= self.init_size:
                self.tree.full_dependency_update()
                self.initialized = True
                del self._n_samples_seen

    def cluster(self):
        tree_data = self.tree.data
        density_mask = tree_data['density'] > self.outlier_threshold
        parents = tree_data['parent']

        # TODO: Probably implement a union-find architecture here
        

        


    def predict(self, X):
        return 


class DPTree:
    def __init__(self):
        raise NotImplementedError("The DPTree class has not yet been implemented")

    def insert(self, cluster_cells):
        # TODO: Code to add previous outlier cells that have increased in density to the tree
        raise NotImplementedError("The insert method has not yet been implemented")

    def delete(self, indices):
        # TODO: Code to delete cluster-cells that have decayed from the tree
        raise NotImplementedError("The delete method has not yet been implemented")



