import numpy as np
from scipy.spatial.distance import cdist

class DPTree:
    """A class to manage the data for cluster cells and the dependency tree."""
    def __init__(self, ndim=48, initial_size=128):
        self.ndim = ndim
        self.dtype = np.dtype([
            ('seed', np.float64, (ndim,)),
            ('density', np.float64),
            ('distance', np.float64),
            ('parent', np.int32),
        ])
        self.size = initial_size
        self._data = np.zeros(self.size, dtype=self.dtype)
        self._in_use = np.zeros(self.size, dtype=bool)
        self._to_update = []
        self.num_cluster_cells = 0

    @property
    def data(self):
        # NOTE: This is a copy, updates will not be reflected in the DP-Tree
        return self._data[self._in_use]

    @property
    def ids(self):
        return np.nonzero(self._in_use)[0]

    @property
    def seeds(self):
        return self._data['seed'][self._in_use]
    
    @property
    def densities(self):
        return self._data['density'][self._in_use]

    def __getitem__(self, key):
        if self._in_use[key]:
            return self._data[key]
        else:
            raise IndexError("Invalid cluster cell id.")

    def get_by_idx(self, indices):
        return self.data[indices]

    def add(self, seeds, densities):
        """Add new cluster-cells, represented as arrays of seeds and densities,
        into the DP-Tree. Potentially dependent cells are marked for update.
        """
        # Check input sizes
        num_new_cells = len(seeds)
        if num_new_cells != len(densities):
            raise ValueError("The number of seeds and densities must match.")

        # Check if we need to resize the array
        if self.num_cluster_cells + num_new_cells > self.size:
            self._resize_array(num_new_cells)

        # Find the first available ids and add the new cells
        new_ids = np.nonzero(~self._in_use)[0][:num_new_cells]
        self._data['seed'][new_ids] = seeds
        self._data['density'][new_ids] = densities
        self._in_use[new_ids] = True
        self.num_cluster_cells += num_new_cells

        # Mark dependent cells for recomputation
        self._to_update.extend(new_ids)
        max_density = np.max(densities)
        is_lower_density = np.less(self._data['density'], max_density)
        self._to_update.extend(np.nonzero(is_lower_density * self._in_use)[0])

        return new_ids

    def remove(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        self._in_use[ids] = False
        self.num_cluster_cells -= len(ids)

        # Mark dependent cells for recomputation
        is_child = np.isin(self._data['parent'], ids)
        self._to_update.extend(np.nonzero(is_child * self._in_use)[0])

    def update_dependencies(self, update_filter=False):
        """Update the dependencies in the tree based on the given update filter
        and as previously marked during addition or removal of cluster cells.
        """
        update_filter = np.isin(self.ids, self._to_update) | update_filter
        seeds = self._data['seed'][self._in_use]
        densities = self._data['density'][self._in_use]

        # (Re)compute the closest cell with a higher density
        less_density = np.less_equal.outer(densities, densities[update_filter])
        distances = cdist(seeds, seeds[update_filter])
        distances[less_density] = np.inf
        closest_ind = np.argmin(distances, axis=0)
        parents = self.ids[closest_ind]
        dependent_dists = distances[closest_ind, np.arange(len(closest_ind))]

        # Set the parent ids and distances
        parents = np.where(dependent_dists == np.inf, -1, parents)
        full_update_filter = np.zeros(self.size, dtype=bool)
        full_update_filter[self._in_use] = update_filter
        self._data['parent'][full_update_filter] = parents
        self._data['distance'][full_update_filter] = dependent_dists

        self._to_update = []

    # TODO: Check if it makes sense to compute update filters for batch learning
    #       by checking the computation time for a full update of the tree
    def full_dependency_update(self):
        """Recompute all cluster-cell dependencies in the tree."""
        seeds = self._data['seed'][self._in_use]
        densities = self._data['density'][self._in_use]

        # (Re)compute the closest cell with a higher density
        less_density = np.less_equal.outer(densities, densities)
        distances = cdist(seeds, seeds)
        distances[less_density] = np.inf
        closest_ind = np.argmin(distances, axis=0)
        parents = self.ids[closest_ind]
        dependent_dists = distances[closest_ind, np.arange(len(closest_ind))]

        # Set the parent ids and distances
        parents = np.where(dependent_dists == np.inf, -1, parents)
        self._data['parent'][self._in_use] = parents
        self._data['distance'][self._in_use] = dep_dist
        
        self._to_update = []
        
    def _resize_array(self, extra_space=0):
        new_size = self.size * 2 + extra_space
        new_data = np.zeros(new_size, dtype=self.dtype)
        new_in_use = np.zeros(new_size, dtype=bool)

        new_data[:self.size] = self._data
        new_in_use[:self.size] = self._in_use

        self._data = new_data
        self._in_use = new_in_use
        self.size = new_size

    def cluster(self, density_threshold):
        assert len(self._to_update) == 0, "Please update dependencies before clustering"
        is_dense = (self._data['density'] > density_threshold) * self._in_use
        cluster_ids = np.nonzero(is_dense)[0]
        parents = self._data[is_dense]['parent']
        not_clustered = (parents != -1)
        while np.any(not_clustered):
            cluster_ids[not_clustered] = parents[not_clustered]
            parents = self._data['parent'][cluster_ids]
            not_clustered = (parents != -1)
        return cluster_ids


