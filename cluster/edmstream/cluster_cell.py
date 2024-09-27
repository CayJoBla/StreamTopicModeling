import numpy as np
import uuid
from collections import Counter
from typing import Callable, List, Optional


class ClusterCell:
    def __init__(
        self, 
        seed: np.ndarray, 
        timestamp: float, 
        decay_fn: Callable, 
        term_threshold: float,
        document: Optional[List[str]] = None
    ):
        """Initialize a new cluster-cell.
        
        Args:
            seed (np.ndarray): The seed point for the cluster-cell.
            timestamp (float): The current timestamp, used to track various
                time-dependent properties of the cluster-cell.
            document (list(str)): A list of words representing a document. This 
                is used for tracking the c-TF-IDF values for the cluster-cell.
        """
        # Initialize cell properties
        self.id = uuid.uuid4()
        self.seed = seed
        self._density = 1

        # Initialize time properties
        self.t_update_density = timestamp
        self.t_update_tf = timestamp
        self.t_insert = timestamp
        self.decay_fn = decay_fn

        # Initialize DPTree properties
        self.dependency = None
        self.dependent_dist = np.inf

        # Term frequency values
        self.term_threshold = term_threshold
        self._tf = Counter(document)

    def __repr__(self):
        return f"ClusterCell({str(self.id)})"

    def _decay_density(self, timestamp):
        """Applies time decay to the density of the cluster-cell."""
        # Check for a proper timestamp
        if timestamp < self.t_update_density:
            raise ValueError("Cannot compute the density at a past timestamp.")
        elif timestamp == self.t_update_density:
            return

        # Decay the density to the current timestamp
        decay_factor = self.decay_fn(timestamp - self.t_update_density)
        self._density *= decay_factor

        # Update timestamp
        self.t_update_density = timestamp

    def get_density(self, timestamp):
        """Returns the density of the cluster-cell at the given timestamp."""
        self._decay_density(timestamp)
        return self._density

    def _decay_tf(self, timestamp):
        """Applies time decay to the term frequency values of the cluster-cell
        and returns the terms that have fallen below the term threshold.
        """
        # Check for a proper timestamp
        if timestamp < self.t_update_tf:
            raise ValueError("Cannot compute TF values at a past timestamp.")
        elif timestamp == self.t_update_tf:
            return []

        # Decay the TF-values to the current timestamp
        decay_factor = self.decay_fn(timestamp - self.t_update_tf)
        below_threshold = []
        for term in self._tf.keys():
            self._tf[term] *= decay_factor
            if (self._tf[term] < self.term_threshold):
                below_threshold.append(term)

        # Update timestamp
        self.t_update_tf = timestamp

        return below_threshold

    def get_tf(self, timestamp):
        """Returns the term frequency values of the cluster-cell at the given 
        timestamp. Removes any terms that have fallen below the term threshold.
        """
        below_threshold = self._decay_tf(timestamp)
        for term in below_threshold:
            del self._tf[term]
        return self._tf
        
    def insert_one(self, timestamp, document=None):
        """Inserts a point into the cluster-cell at the specified timestamp.
        
        Args:
            timestamp (float): The current timestamp, used to update the 
                density of the cluster-cell before new points are inserted
            document (list(str)): A list of words representing a document.
        """
        # Update the density of the cluster-cell
        self._decay_density(timestamp)
        self._density += 1

        # Update the c-TF-IDF values
        if document is not None:
            below_threshold = self._decay_tf(timestamp)
            self._tf.update(document)

            for term in below_threshold:
                if self._tf[term] < self.term_threshold:
                    del self._tf[term]

        self.t_insert = timestamp   # Update the last insert time

