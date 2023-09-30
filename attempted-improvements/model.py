import torch
import numpy as np

class HNSW:
    def __init__(self, max_elements, dim, M=16, ef_construction=200, random_seed=None):
        """
        Initialize the HNSW structure.
        
        Parameters:
            - max_elements (int): Maximum number of elements in the structure.
            - dim (int): Dimensionality of the vector space.
            - M (int): Maximum number of bi-directional links created for every new element during construction.
            - ef_construction (int): Size of the dynamic list for the nearest neighbors during the construction.
            - random_seed (int or None): Seed for random number generator.
        """
        self.max_elements = max_elements
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        
        # Validate input parameters
        if max_elements <= 0:
            raise ValueError("max_elements must be positive.")
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if M <= 0:
            raise ValueError("M must be positive.")
        if ef_construction <= 0:
            raise ValueError("ef_construction must be positive.")
        
        # Random number generator
        self.rng = np.random.default_rng(seed=random_seed)
        
        # Layers
        self.layers = [torch.empty((0, dim))]
        
        # Friends at each layer
        self.friends = [torch.empty((0, M, dtype=torch.long)])


    def distance(self, a, b):
        """
        Compute the distance between two points.
        
        Parameters:
            - a (tensor): The first point.
            - b (tensor): The second point.
        
        Returns:
            - float: The distance between the two points.
        """
        # Validate input tensors
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            raise TypeError("Both points should be torch tensors.")
        if a.size() != b.size():
            raise ValueError("Both points should have the same dimensions.")
        
        return torch.norm(a - b)

    def insert(self, point):
        """
        Insert a point into the HNSW structure.
        
        Parameters:
            - point (list or tensor): The point to be inserted.
        """
        # Validate input point
        if not isinstance(point, (list, torch.Tensor)):
            raise TypeError("Point should be a list or a torch tensor.")
        
        point = torch.tensor(point)
        # Insertion code here
        # ...

    def search(self, query, ef_search=50):
        """
        Search for the nearest neighbors of the query point.
        
        Parameters:
            - query (list or tensor): The query point.
            - ef_search (int): The size of the dynamic list for nearest neighbors during searching.
        
        Returns:
            - list: The indices of the nearest neighbors.
        """
        # Validate input query
        if not isinstance(query, (list, torch.Tensor)):
            raise TypeError("Query should be a list or a torch tensor.")
        
        query = torch.tensor(query)
        # Search code goes here
        # ...

    def _search_layer(self, query, layer, max_candidates, visited):
        """
        Navigable search at a specific layer.
        
        Parameters:
            - query (tensor): The query point.
            - layer (int): The layer index.
            - max_candidates (int): The maximum number of candidates to consider.
            - visited (set): Set of already visited nodes.
        
        Returns:
            - list: The indices of the nearest neighbors in the specified layer.
        """
        # Validate input parameters
        if not torch.is_tensor(query):
            raise TypeError("Query should be a torch tensor.")
        if not isinstance(layer, int) or layer < 0:
            raise ValueError("Layer should be a non-negative integer.")
        if not isinstance(max_candidates, int) or max_candidates <= 0:
            raise ValueError("max_candidates should be a positive integer.")
        if not isinstance(visited, set):
            raise TypeError("Visited should be a set.")
        
        # Navigable search at a specific layer
        # ...

    def _link(self, point_index, layer_index):
        """
        Link a point at a specific layer.
        
        Parameters:
            - point_index (int): The index of the point to be linked.
            - layer_index (int): The index of the layer where the point will be linked.
        """
        # Validate input parameters
        if not isinstance(point_index, int) or point_index < 0:
            raise ValueError("point_index should be a non-negative integer.")
        if not isinstance(layer_index, int) or layer_index < 0:
            raise ValueError("layer_index should be a non-negative integer.")
        
        # Linking point at a specific layer
        # ...

    def _get_entering_point(self, layer_index):
        """
        Get a random entering point in a specific layer.
        
        Parameters:
            - layer_index (int): The index of the layer.
        
        Returns:
            - tensor: The entering point.
        """
        # Validate input parameters
        if not isinstance(layer_index, int) or layer_index < 0:
            raise ValueError("layer_index should be a non-negative integer.")
        
        # Get a random entering point in a specific layer
        # ...

    def __str__(self):
        """
        String representation of the HNSW structure.
        
        Returns:
            - str: A string representation of the HNSW structure.
        """
        return f"HNSW(max_elements={self.max_elements}, dim={self.dim}, M={self.M}, ef_construction={self.ef_construction})"
