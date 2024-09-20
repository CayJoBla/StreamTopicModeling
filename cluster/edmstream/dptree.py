import numpy as np


class DPTree:
    def __init__(self, num_initial_cells):
        self._cluster_cells = dict()
        self._sorted_ids = []
        self.initialized = False
        self.num_initial_cells = num_initial_cells

    def __len__(self):
        return len(self._cluster_cells)

    def keys(self):
        return self._cluster_cells.keys()

    def values(self):
        return self._cluster_cells.values()

    def items(self):
        return self._cluster_cells.items()

    def __getitem__(self, cell_id):
        return self._cluster_cells[cell_id]

    @property
    def cluster_cells(self):
        return [self._cluster_cells[cell_id] for cell_id in self._sorted_ids]

    @property
    def ids(self):
        return self._sorted_ids

    def initialize(self, timestamp):
        """Initialize the DP-Tree by sorting the cluster-cells by density.

        Parameters:
            timestamp (float): The current timestamp, used to determine the 
                current timely-density of cluster-cells.
        """
        # Check if the tree has enough cells to initialize
        if len(self._cluster_cells) == 0:
            raise ValueError("The DP-Tree cannot be initialized without any "
                                "active cluster-cells.")

        # Sort the cluster-cells by density
        self._sorted_ids = sorted(
            self._cluster_cells.keys(), 
            key=lambda q: self._cluster_cells[q].get_density(timestamp)
        )

        # Determine cluster-cell dependencies and build the tree
        cluster_cells = self.cluster_cells
        seeds = np.array([cell.seed for cell in cluster_cells])
        for i, cluster_cell in enumerate(cluster_cells[:-1]):
            distances = np.linalg.norm(seeds[i+1:] - seeds[i], axis=1)
            closest_idx = np.argmin(distances)
            cluster_cell.dependency = self._sorted_ids[closest_idx+(i+1)]
            cluster_cell.dependent_dist = distances[closest_idx]

        self.initialized = True

    def insert(self, cluster_cell, timestamp):
        """Add a new cluster-cell to the DP-Tree while maintaining the density-
        sorted ordering of the cluster-cells. Automatically updates dependencies
        where necessary.

        Args:
            cluster_cell (ClusterCell): The cluster-cell to insert into the tree
            timestamp (float): The current timestamp, used to determine the 
                current timely-density of cluster-cells.
            
        Returns:
            (bool): Whether the insertion caused any dependency updates within
                the existing tree.
        """
        # Reset dependency and insert the new cluster-cell
        cluster_cell.dependency = None
        cluster_cell.dependent_dist = np.inf
        self._cluster_cells[cluster_cell.id] = cluster_cell

        # Don't bother sorting until the tree is initialized
        if not self.initialized:
            self._sorted_ids.append(cluster_cell.id)
            if len(self._cluster_cells) >= self.num_initial_cells:
                self.initialize(timestamp)
            return self.initialized
        
        # Insert the new cluster-cell into the sorted list
        density = cluster_cell.get_density(timestamp)
        for idx, cell_id in enumerate(self._sorted_ids):
            if density <= self._cluster_cells[cell_id].get_density(timestamp):
                break
        else:
            idx += 1
        self._sorted_ids.insert(idx, cluster_cell.id)

        # Update the dependencies and dependent distances of the cluster-cells
        return self.update_dependencies(idx, self._sorted_ids[:idx])

    def remove_outliers(self, timestamp, density_threshold):
        """Remove any cluster-cells whose densities have fallen below the 
        density threshold. Returns the list of outlier cluster-cells that were 
        removed.

        Args:
            timestamp (float): The current timestamp, used to determine the 
                current timely-density of cluster-cells.
            density_threshold (float): The density value at which cluster-cells 
                are considered outliers and are removed.

        Returns:
            (list): A list of the outlier cluster-cells that were removed.
        """
        if not self.initialized:
            return []  # Ignore outliers until the tree is initialized

        # Find the first cluster-cell that satisfies the density threshold
        for idx, cell_id in enumerate(self._sorted_ids):
            cluster_cell = self._cluster_cells[cell_id]
            if cluster_cell.get_density(timestamp) > density_threshold:
                break
        
        # Remove and return the outlier cluster-cells
        self._sorted_ids = self._sorted_ids[idx:]
        if len(self._sorted_ids) == 0:
            self.initialized = False    # Reset the tree if no active cells
        return [self._cluster_cells.pop(cell_id) 
                for cell_id in self._sorted_ids[:idx]]

    def assign_point_to_cell(self, cell_id, timestamp, cell_distances, 
                             document=None):
        """Assigns a new point to the cluster-cell with the given id.
        Automatically updates the sorted list of cluster-cell ids and updates 
        dependencies according to the density and triangle inequality filters.
        """
        # Update the density of the cluster-cell
        cluster_cell = self._cluster_cells[cell_id]
        cluster_cell.insert_one(timestamp, document)

        if not self.initialized:
            return False    # Ignore dependencies until the tree is initialized

        # Get the index of the updated cluster-cell in the sorted list
        sorted_idx = self._sorted_ids.index(cell_id)
        greater_density_order = self._sorted_ids[sorted_idx+1:].copy()
        
        new_density = cluster_cell.get_density(timestamp)
        to_update = []
        idx = sorted_idx
        for current_cell_id in greater_density_order:
            idx += 1

            # Density Filter
            current_cell = self._cluster_cells[current_cell_id]
            current_density = current_cell.get_density(self.timestamp)
            if new_density <= current_density:
                idx -= 1  
                break       # Filter satisfied, no need to check further

            # Bubble sort: _sorted_ids[idx] should become the updated cell
            self._sorted_ids[idx], self._sorted_ids[idx-1] = \
                self._sorted_ids[idx-1], self._sorted_ids[idx]

            # Tri-Ineq Filter 
            difference = abs(cell_distances[idx]-cell_distances[sorted_idx])
            if difference <= current_cell.dependent_dist:
                to_update.append(current_cell_id)

        # Update the dependencies and dependent distances of the cluster-cells
        return self.update_dependencies(idx, to_update)

    def update_dependencies(self, sorted_cell_idx, to_update):
        """Update dependencies and dependent distances as a cause of a 
        cluster-cell update/insert

        Args:
            sorted_cell_idx (int): The new index of the updated/inserted 
                cluster-cell in the sorted list of cluster-cell ids
            to_update (list): A list of the cluster-cell ids that need to be 
                checked for dependency updates. These cluster-cells should all 
                have a lower density than the updated cluster-cell.

        Returns:
            (bool): Whether the cell update caused any dependency changes within 
                    the existing tree (cell insertion always returns True).
        """
        cell_id = self._sorted_ids[sorted_cell_idx]
        cluster_cell = self._cluster_cells[cell_id]
        caused_update = False 

        if len(to_update) != 0:
            # Compute distances to the updated cell for dependency updates
            seeds = np.array([self._cluster_cells[cell_id].seed 
                                for cell_id in to_update])
            distances = np.linalg.norm(seeds - cluster_cell.seed, axis=1)

            # Check whether the updated cluster-cell has any dependents
            for idx, cell_id in enumerate(to_update):
                current_cluster_cell = self._cluster_cells[cell_id]
                if distances[idx] < current_cluster_cell.dependent_dist:
                    caused_update = True
                    current_cluster_cell.dependency = cluster_cell.id
                    current_cluster_cell.dependent_dist = distances[idx]

        # Compute the dependency of the updated cluster-cell
        if sorted_cell_idx == (len(self._sorted_ids)-1):
            # If the updated cell is the highest density, it has no dependencies
            cluster_cell.dependency = None
            cluster_cell.dependent_dist = np.inf
        elif (cluster_cell.dependency in to_update) or \
                (cluster_cell.dependency is None):
            # Only update dependency if the current dependency is now a lower
            # density (in the to_update list) or if the current cell is new
            greater_ids = self._sorted_ids[sorted_cell_idx+1:]
            greater_seeds = np.array([self._cluster_cells[cell_id].seed 
                                        for cell_id in greater_ids])
            greater_dists = np.linalg.norm(greater_seeds - cluster_cell.seed, 
                                            axis=1)
            min_idx = np.argmin(distances)
            cluster_cell.dependency = greater_ids[min_idx]
            cluster_cell.dependent_dist = greater_dists[min_idx]
            caused_update = True

        return caused_update

    def query_subtrees(self, distance_threshold):
        """Query the root dependencies of all cluster-cells in the tree."""
        if not self.initialized:
            raise ValueError("The DP-Tree has not yet been initialized")

        labels = dict()
        clusters = dict()
        for cell_id in reversed(self._sorted_ids):
            cluster_cell = self._cluster_cells[cell_id]
            if cluster_cell.dependent_dist > distance_threshold:
                labels[cell_id] = cell_id
                clusters[cell_id] = [cell_id]
            else:
                labels[cell_id] = labels[cluster_cell.dependency]
                clusters[labels[cell_id]].append(cell_id)

        return clusters, labels