import copy
import math
from collections import deque
from typing import List
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.sparse import csr_matrix, csgraph
import networkx as nx
from scipy.sparse import csr_matrix
# from bertopic.vectorizer import ClassTfidfTransformer

# TODO: As soon as I implement delete in the DPTree, all indexing will be messed
#       up, and I will need to come up with a new way to index the cluster-cells

class EDMStream:
    """My custom implementation of the EDMStream algorithm. 

    See the original technical paper for more details:
    "Clustering Stream Data by Exploring the Evolution of Density Mountain"
    (https://arxiv.org/pdf/1710.00867)
    """
    def __init__(
        self,
        decay_alpha=0.98,
        decay_lambda=1.,
        beta=4e-3,
        radius=0.1,
        batch_size=128,
        init_size=1024,
    ):
        super().__init__()
        self.timestamp = 0
        self.decay_factor = decay_alpha ** decay_lambda

        self._n_samples_seen = 0
        self.n_clusters = 0

        self.outlier_threshold = beta * v / (1-self.decay_factor)

        self.batch_size = batch_size
        self.init_size = initialization_size
        self.initialized = False
        self.init_buffer = list()

        # TODO: Check that parameters are within the valid ranges
        # TODO: Initialize needed attributes (ex. stream speed, time period, etc.)

    # TODO: If the storage structure of the cluster-cells changes, this needs to be updated
    @property
    def cluster_cell_seeds(self):
        return np.array([cell.seed for cell in self.cluster_cells])
    
    @property
    def cluster_cell_densities(self):
        return np.array([cell.density for cell in self.cluster_cells])

    def get_update_filter(self, old_densities, distances, point_assignments):
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
        is_updated = new_densities > old_densities
        print("Updated Cells Mask:", is_updated, sep="\n", end="\n\n")
        
        if not np.any(is_updated):  # No updated densities
            return np.zeros_like(is_updated, dtype=bool)

        # Compute the density filter
        new_densities = self.cluster_cell_densities
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
        for cell_idx in np.where(is_update_cause)[0]:  
            print("------------------------------------------------")
            print("Cell Index:", cell_idx, end="\n\n")

            point_mask = (point_assignments == cell_idx)
            print("Assigned Point Mask:", point_mask, sep="\n", end="\n\n")

            differences = np.abs(np.subtract(
                distances[density_filter, None,     point_mask],
                distances[None,           cell_idx, point_mask]
            ))
            print("Differences:", differences, sep="\n", end="\n\n")

            not_tri_ineq = (differences <= dep_distances[needs_update,None])
            print("Not Tri-Ineq:", not_tri_ineq, sep="\n", end="\n\n")

            tri_ineq_filter |= np.any(not_tri_ineq, axis=-1).ravel()
            print("Triangle Inequality Filter:", tri_ineq_filter, sep="\n", end="\n\n")

        return needs_update

    def create_outlier_cells(self, X):
        """Adds new cluster-cells from a batch of outlier points.
        This implementation is greedy, choosing the point with the most 
        neighbors as the seed for the new cluster-cell at each iteration.

        Point assignments and new cluster-cells are returned.
        """
        # Which points are within the radius of each other
        close_points = (cdist(X,X) <= self.r)

        new_active_cells = list()
        assignments = np.full(len(X), -1, dtype=int)
        current_cell_idx = len(self.cluster_cells)
        while np.any(close_points):
            # Find the point with the most neighbors
            idx = np.argmax(close_points.sum(axis=1))
            members = np.where(close_points[idx])[0]
            density = len(members)

            # Create a new cluster-cell and assign the points
            new_cell = ClusterCell(
                seed = X[idx], 
                density = density,
                timestamp = self.timestamp, 
            )
            new_cluster_cells.append(new_cell)
            assignments[members] = current_cell_idx
            current_cell_idx += 1
            close_points[members], close_points[:,members] = False, False

        return assignments, new_active_cells

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
        X = np.atleast_2d(X)
        num_points = X.shape[0]
        num_cells = len(self.cluster_cells)

        # Assign points to existing cluster-cells
        assignments = np.full(num_points, -1, dtype=int)
        if num_cells > 0:
            # Try to insert the points into existing cluster-cell
            seeds = self.cluster_cell_seeds
            distances = cdist(seeds, X)
            closest_cell_idx = np.argmin(distances, axis=0)
            closest_cell_dist = distances[closest_cell_idx]
            inlier_mask = closest_cell_dist <= self.r

            # TODO: Check which of these 4 methods is fastest

            # point_mask = np.equal.outer(np.arange(num_cells), closest_cell_idx)
            # point_mask *= inlier_mask
            # np.sum(point_mask, axis=1) # TODO: Add these densities to the cluster-cells

            for idx, cell in enumerate(self.cluster_cells):
                point_mask = inlier_mask * (closest_cell_ind == idx)
                cell.insert(np.sum(point_mask), self.timestamp)

            # for cell_idx in closest_cell_ind[inlier_mask]:
            #     self.cluster_cells[cell_idx].insert(1, self.timestamp)

            # for cell_idx, num_added in zip(np.unique(closest_cell_ind[inlier_mask], return_counts=True)):
            #     self.cluster_cells[cell_idx].insert(num_added, self.timestamp)

            assignments[inlier_mask] = closest_cell_idx
            outlier_mask = ~inlier_mask
        else:
            distances = np.array([])
            outlier_mask = np.ones(num_points, dtype=bool)
        
        # Assign points to outlier cluster-cells
        if np.any(outlier_mask):
            labels, new_cluster_cells = self.create_outlier_cells(X[outlier_mask])
            assignments[outlier_mask] = labels
        else:
            new_cluster_cells = list()

        return assignments, distances, new_cluster_cells

    def learn_batch(self, X):
        """Processes a batch of new points in the stream."""
        # Checks on the input data
        assert X.ndim == 2, "Input data must be 2-dimensional"
        assert X.shape[0] == self.batch_size, "Input batch size must match the batch_size parameter"

        # Compute timestep updates
        self.timestamp += 1     # TODO: Determine the proper timestamp handling
        # TODO: I need to update all densities here (decay)
        # TODO: Check outlier cells for memory space recycling
        pre_update_densities = self.cluster_cell_densities

        # Assign points to cluster-cells and create new cluster-cells
        assignments, distances, new_cluster_cells  = self.assign(X)
        
        # Determine which cluster-cells need to be updated
        if self.initialized and len(self.cluster_cells) > 0:    # TODO: I think all need to be updated if new cells are added
            update_filter = self.get_update_filter(
                pre_update_densities, distances, assignments
            )
        else:
            # NOTE: 2 cases: 1) Not initialized, 2) Initialized, but no cluster-cells (is case 2 possible?)
            if len(new_cluster_cells) > 0:
                self.cluster_cells.extend(new_cluster_cells)
            else:
                raise ValueError("What is happening? No old cluster cells and no new cluster cells")

            # TODO: Determine what the initialization condition is
            if self.initialized: # or if initialization condition is met:
                update_filter = np.ones(len(self.cluster_cells), dtype=bool)

        # TODO: Recompute dependent distances for all marked cluster-cells 
        #       (i.e. those in the update filter)

        

    def predict(self, X):
        return 




class ClusterCell:
    def __init__(self, seed, density, timestamp, documents=None):
        """Initialize a new cluster-cell.
        
        Args:
            seed (np.ndarray): The seed point for the cluster-cell.
            density (int): The initial density of the cluster-cell.
            timestamp (int): The current timestamp, used to track the time
                since outlier cells received new points. If enough time has
                passed, the outlier cells will be removed
            documents (List[List(str)]): A list of documents, where each 
                document is split into a list of its words. This is used to
                compute the c-TF-IDF values for the cluster-cell.
        """
        # Initialize cell properties
        self.t_update = timestamp
        self.t_insert = timestamp
        # TODO: Implement a way to get a unique ID for each cell
        self.seed = seed
        self.density = density

        # c-TF-IDF values
        self.tf = dict()        # TODO: May need to initialize with {"word":0}
        term_threshold = 0.1    # TODO: Determine some theoretical optimal value
        if documents is not None:
            document = sum(documents, [])       # Combine into a single document
            words, counts = np.unique(document, return_counts=True)
            self.tf = {word: count for word, count in zip(words, counts)}

    def decay(self, timestamp):
        """Applies decays to the time-density and term weights of the 
        cluster-cell.
        """
        if timestamp < self.t_update:
            raise ValueError("Cannot update a cluster-cell to a past timestamp")
        elif timestamp == self.t_update:
            return
        self.time_density *= self.fading_function(timestamp - self.t_update)
        self.t_update = timestamp
        
    def insert(self, n):
        """Inserts n points into the cluster-cell.
        
        Args:
            n (int): The number of points to add to the cluster-cell
            timestamp (int): The current timestamp, used to update the time-density
                             of the cluster-cell before new points are inserted
        """
        self._decay(timestamp)
        self.time_density += n


class DPTree:
    def __init__(self):
        raise NotImplementedError("The DPTree class has not yet been implemented")

    def insert(self, cluster_cells):
        # TODO: Code to add previous outlier cells that have increased in density to the tree
        raise NotImplementedError("The insert method has not yet been implemented")

    def delete(self, indices):
        # TODO: Code to delete cluster-cells that have decayed from the tree
        raise NotImplementedError("The delete method has not yet been implemented")



