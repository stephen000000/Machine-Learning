import numpy as np
from math import log
from node import Node

class C45:
    """Creates a C4.5 decision tree"""
    def __init__(self, X_train, y_train):
        self.train_set = np.c_[X_train.values, y_train.values]                      # combines the training set and target set into a single dataset (numpy array)                              
        self.features = list(X_train.columns)                                       # list of features
        self.num_features = len(self.features)                                      # number of features
        self.feature_num_type = self.get_num_types(self.features, self.train_set)   # features numeric types i.e. discrete or continuous (dictionary)
        self.classes = self.get_classes(self.train_set[:, -1])                      # list of classes
        self.tree = None                                                            # root node of the tree (node object)
        self.max_depth = 0

    """Returns the numeric types for the features in a dictionary"""
    def get_num_types(self, features, data):
        num_types = {}

        # Checks to see if a feature is continuous or discrete on the first value
        for i in range(len(features)):
            if isinstance(data[0][i], int):
                num_types[features[i]] = "discrete"
            elif isinstance(data[0][i], float):
                num_types[features[i]] = "continuous"

        return num_types

    """Returns the different classes in a list"""
    def get_classes(self, target):
        classes = []
        classes.append(target.item(0))
        
        for i in range(1, len(target)):
            t = target.item(i)

            if t not in classes:
                classes.append(t)

        return classes

    """Calls the recursive function to generate the tree"""
    def generate_tree(self, pruning=True):
        self.tree = self.generate_tree_recursive(self.train_set, self.features, -1, pruning)

    """Recursive function to generate the tree"""
    def generate_tree_recursive(self, data, features, depth, pruning):
        depth += 1
        if depth > self.max_depth:
            self.max_depth = depth

        target = data[:, -1]
        is_leaf = self.check_same_class(target)
        maj_class = self.get_maj_class(target)
        
        if len(data) == 0:
            return Node("Fail", None, True, depth)
        elif is_leaf is True:
            return Node(target[0], None, True, depth)
        elif pruning is True and maj_class is not False:
            return Node(maj_class, None, True, depth)
        else:
            (parent, children) = self.split_node(data, features)
            parent.depth = depth
            
            for c in children:
                parent.child_nodes.append(self.generate_tree_recursive(np.array(c), features, depth, pruning))

            return parent
    
    """Checks if the target values are all in the same class"""
    def check_same_class(self, target):
        t1 = target[0]

        for t in target:
            if t != t1:
                return False
        else:
            return True

    def get_maj_class(self, target):
        num_each_class = {}
        for c in self.classes:
            num_each_class[c] = 0

        for t in target:
            num_each_class[t] += 1

        cut_off = 0.9

        for k, v in num_each_class.items():
            if v/len(target) >= cut_off:
                return k
        return False

    """Splits the node based on the feature value with the best info gain.
       Returns the parent node and a list of lists of the children nodes"""
    def split_node(self, data, features):
        best_feature, best_threshold, best_branches = None, None, []
        best_info_gain = -1 * float("inf")

        def split_with_discrete():
            print("Discrete")

        def split_with_continuous(data, features, feat_index):
            nonlocal best_feature, best_threshold, best_branches, best_info_gain
            data = data[data[:, feat_index].argsort()]
            
            for j in range(0, len(data) - 1):       
                if data[j][-1] != data[j+1][-1]:
                    threshold = (data[j][feat_index] + data[j+1][feat_index])/2
                    left_branch = []
                    right_branch = []

                    for k in range(len(data)):
                        if(k < j+1):
                            left_branch.append(data[k])
                        else:
                            right_branch.append(data[k])
                    
                    info_gain = self.calc_info_gain(data, [left_branch, right_branch])

                    if info_gain > best_info_gain:
                        best_feature = features[feat_index]
                        best_threshold = threshold
                        best_branches = [left_branch, right_branch]
                        best_info_gain = info_gain

        for i in range(len(features)):
            ntype = self.feature_num_type[features[i]]

            if ntype is "discrete":
                split_with_discrete()
            elif ntype is "continuous":
                split_with_continuous(data, features, i)

        return (Node(best_feature, best_threshold, False), best_branches)

    """Returns the information gain"""
    def calc_info_gain(self, data, branches):
        ent_before_split = self.calc_ent(data)

        weights = []
        n_rows = len(data)
        for b in branches:
            weights.append(len(b)/n_rows)

        ent_after_split = 0
        for i in range(len(weights)):
            ent_after_split += weights[i] * self.calc_ent(branches[i])

        return ent_before_split - ent_after_split

    """Returns the entropy"""
    def calc_ent(self, data):
        props = self.calc_proportions(data)
        
        ent = 0
        for n in props.values():
            if n == 0:
                ent -= 0
            else:
                ent -= n * log(n, 2)

        return ent

    """Returns the proportion of each class in the dataset"""
    def calc_proportions(self, data):
        num_each_class = {}
        for c in self.classes:
            num_each_class[c] = 0

        for r in data:
            num_each_class[r[-1]] += 1

        n_rows = len(data)
        for k, v in num_each_class.items():
            num_each_class[k] = v/n_rows

        return num_each_class

    """Calls the recursive function to print the tree"""
    def print_tree(self):
        self.print_tree_recursive(self.tree)

    """Recursive function to print the tree"""
    def print_tree_recursive(self, node, indent=""):
        if not node.is_leaf:
            if node.best_threshold is None:
				#discrete
                for index,child in enumerate(node.child_nodes):
                    if child.isLeaf:
                        print(indent + node.best_feature + " = " + attributes[index] + " : " + child.best_feature)
                    else:
                        print(indent + node.best_feature + " = " + attributes[index] + " : ")
                        self.print_tree_recursive(child, indent + "	")
            else:
                left_child = node.child_nodes[0]
                right_child = node.child_nodes[1]

                if left_child.is_leaf:
                    print(indent + node.best_feature + " < " + str(node.best_threshold) + " : " + left_child.best_feature)
                else:
                    print(indent + node.best_feature + " < " + str(node.best_threshold)+" : ")
                    self.print_tree_recursive(left_child, indent + "	")

                if right_child.is_leaf:
                    print(indent + node.best_feature + " => " + str(node.best_threshold) + " : " + right_child.best_feature)
                else:
                    print(indent + node.best_feature + " => " + str(node.best_threshold) + " : ")
                    self.print_tree_recursive(right_child , indent + "	")

    def predict(self, X_test, y_test):
        test_set = X_test.values
        target = y_test.values
        success = 0
        total = len(test_set)

        for i in range(total):
            if self.tree == None:
                print("No tree generated.\nCall the generate() function.")
                return
            else:
                result = self.predict_recursive(self.tree, test_set[i])

            if result == target[i]:
                success += 1

        return success/total * 100

    def predict_recursive(self, node, test_row):
        result = None

        if node.is_leaf:
            return node.best_feature
        else:
            i = self.features.index(node.best_feature)
            feature_val = test_row[i]

            if feature_val <= node.best_threshold:
                result = self.predict_recursive(node.child_nodes[0], test_row)
            else:
                result = self.predict_recursive(node.child_nodes[1], test_row)

            return result

    """Prints details about the tree.
       Used for debugging"""
    def prt(self):
        print("Data:", self.train_set)
        print("Features:", self.features)
        print("Number of Features:", self.num_features)
        print("Feature Numeric Types:", self.feature_num_type)
        print("Classes:", self.classes)