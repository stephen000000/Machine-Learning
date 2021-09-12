class Node:
    def __init__(self, best_feature, best_threshold, is_leaf, depth = None):
        self.best_feature = best_feature
        self.best_threshold = best_threshold
        self.is_leaf = is_leaf
        self.depth = depth
        self.child_nodes = []