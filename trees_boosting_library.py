# implement DecisionTree with a fit and predict method
# Set up the skeleton for the DecisionTree class

import numpy as np
from sklearn.exceptions import NotFittedError
import copy

np.random.seed(42)

class DecisionTree:
    def __init__(self, criterion='misclassification', max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Base classifier for a decision tree, that will be used for random forest and boosting.

        Parameters
        ----------
        criterion: str
            Either 'misclassification', 'gini', or 'entropy'.
        max_depth: int
            The maximum depth the tree should grow.
        min_samples_split: int
            The minimum number of samples required to split.
        min_samples_leaf: int
            The minimum number of samples required for a leaf node.
        """    
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    
    def fit(self, X, y):
        """
        Fit the decision tree classifier.

        Parameters
        ----------
        X: numpy.ndarray
            The training data of size (n_samples, n_features).
        y: numpy.ndarray
            The labels of size (n_samples,).
        """
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))    

        self.tree = self._grow_tree(X, y, depth=0)

        return self

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively 'grows' a decision tree.
        
        Parameters
        ----------
        X: numpy.ndarray
            The training data of size (n_samples, n_features).
        y: numpy.ndarray
            The labels of size (n_samples,).
        depth: int
            The current depth of the tree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping conditions
        if(self.max_depth is not None and depth >= self.max_depth or
                n_samples < self.min_samples_split or 
                n_labels == 1):
            
            prediction = np.argmax(np.bincount(y))
            return Node(is_leaf = True, prediction = prediction) 

        # Find the best split
        feature_idx, threshold = self._best_split(X, y)

        # No split found, create a leaf node
        if feature_idx is None:
            prediction = np.argmax(np.bincount(y))
            return Node(is_leaf = True, prediction = prediction)

        # Split the data
        feature_values = X[:, feature_idx]
        left_idx = np.where(feature_values <= threshold)
        right_idx = np.where(feature_values > threshold)

        # Check min_samples_leaf constraint
        if len(left_idx[0]) < self.min_samples_leaf or len(right_idx[0]) < self.min_samples_leaf:
            prediction = np.argmax(np.bincount(y))
            return Node(is_leaf = True, prediction = prediction)

        # Recursively grow left and right subtree
        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(feature_idx = feature_idx, threshold = threshold, left = left, right = right)
    
    def _best_split(self, X, y):
        """
        Find the best split for a node.
        
        Parameters
        ----------
        X: numpy.ndarray
            The training data of size (n_samples, n_features).
        y: numpy.ndarray
            The labels of the node.
        
        Returns
        -------
        tuple
            The index of the feature to split and the threshold to split on.
        """
        m = X.shape[0]
        if m <= 1:
            return None, None
    
        # Calculate the impurity for the parent
        parent_impurity = self._node_impurity(y)

        # Initialize the best split
        best_info_gain = -float('inf')
        best_feature_idx = None
        best_threshold = None

        for feature_idx in range(self.n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                # Split the node
                left_idx = np.where(feature_values <= threshold)
                right_idx = np.where(feature_values > threshold)

                # if split violates min_samples_leaf constraint
                if len(left_idx[0]) < self.min_samples_leaf or len(right_idx[0]) < self.min_samples_leaf:
                    continue
                    
                # Calculate the impurity
                left_impurity = self._node_impurity(y[left_idx])
                right_impurity = self._node_impurity(y[right_idx])

                # Calculate the weights of the child nodes
                left_weight = len(left_idx[0]) / m
                right_weight = len(right_idx[0]) / m

                # Calculate the information gain
                info_gain = parent_impurity - (left_weight * left_impurity + right_weight * right_impurity)

                # Update if split is better
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def predict(self, X):
        """
        Predict the label of each sample in X. Note this is only for binary classification.
        
        Parameters
        ----------
        X: numpy.ndarray
            The training data of size (n_samples, n_features)."""
        if self.tree is None:
            raise NotFittedError('Estimator not fitted, call `fit` first.')
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverse the tree to find the prediction for a given sample.
        
        Parameters
        ----------
        x: numpy.ndarray
            The sample of size (n_features,).
        node: Node
            The current node being evaluated.
        """
        # If leaf node, return the prediction
        if node.is_leaf is True:
            return node.prediction

        # Traverse left or right subtree
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def _node_impurity(self, y):
        """
        Compute the impurity of a node. 

        Parameters
        ----------

        y: numpy.ndarray
            The labels of the node.
        """

        if self.criterion == 'misclassification':
            miss_rate = 1 - np.max(np.bincount(y) / y.size)
            return miss_rate
        
        elif self.criterion == 'gini':
            gini = 1 - np.sum((np.bincount(y) / y.size) ** 2)
            return gini
        
        elif self.criterion == 'entropy':
            # include epislon to avoid log(0)
            epsilon = 1e-10
            p = np.bincount(y) / len(y)
            entropy = -np.sum(p * np.log2(p + epsilon))
            return entropy
        
        else:
            raise ValueError('Criterion should be "misclassification", "gini", or "entropy"')
        
class Node:
    def __init__(self, is_leaf = False, feature_idx=None, threshold=None, left=None, right=None, prediction=None):
        """
        Node in decision tree
        
        Parameters
        ----------
        is_leaf: bool
            Whether the node is a leaf
        feature_idx: int
            Index of feature to split on
        threshold: float
            Value to split feature on
        left: Node
            Left child node
        right: Node
            Right child node
        Prediction: int or float
            Value at leaf node (class label)
        """
        self.is_leaf = is_leaf
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction 

# Set up skeleton for RandomForest class
class RandomForest():
    def __init__(self, classifier, num_trees, min_features):
        """
        A random forest classifier that uses DecisionTree as the base classifier.
        
        Parameters
        ----------
        classifer: DecisionTree
            The classifier used as a base learner. An object of the DecisionTree class.
        num_trees: int
            The number of trees in the forest.
        min_features: int
            The minimum number of features to consider when looking
            for the best split.
        """
        self.classifier = classifier
        self.num_trees = num_trees
        self.min_features = min_features  
        self.trees = []   
    
    def fit(self, X, y):
        """
        Fit the random forest classifier.

        Parameters
        ----------

        X: numpy.ndarray
            The training data of size (n_samples, n_features).
        y: numpy.ndarray
            The labels of size (n_samples,).
        """
        n_samples, n_total_features = X.shape
        if self.min_features > n_total_features:
            raise ValueError('min_features must be less than or equal to the total number of features')
        
        self.trees = [] # clear out any previous trees
        for _ in range(self.num_trees):
            #bootstrap sample
            sample_indices = np.random.choice(n_samples, size = n_samples, replace = True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            #random subset of features
            n = np.random.randint(self.min_features, n_total_features + 1)
            feature_indices = np.random.choice(n_total_features, size = n, replace=False)

            tree = copy.deepcopy(self.classifier)

            # subset of data
            X_sample_subset = X_sample[:, feature_indices]

            #fit tree
            tree.fit(X_sample_subset, y_sample)

            # store selected feature indices on tree for prediction
            tree.feature_indices = feature_indices
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        """
        Predict the label of each sample in X. Note this is only for binary classification.
        
        Parameters
        ----------
        X: numpy.ndarray
            The training data of size (n_samples, n_features).
        """
        all_tree_predictions = []
        for tree in self.trees:
            X_subset = X[:, tree.feature_indices]
            predictions = tree.predict(X_subset)
            all_tree_predictions.append(predictions)

        all_tree_predictions = np.array(all_tree_predictions)

        # majority vote
        final_predictions = []
        for i in range(all_tree_predictions.shape[1]):
            votes = np.bincount(all_tree_predictions[:, i])
            final_predictions.append(np.argmax(votes))
        return np.array(final_predictions)

class AdaBoost(DecisionTree):
    def __init__(self, weak_learner, num_learners, learning_rate):
        """
        An AdaBoost classifier that uses DecisionTree as the base classifier.
        
        Parameters
        ----------

        weak_learner: DecisionTree
            The classifier used as a weaker learner. An object of the DecisionTree class.
        num_learners: int
            The maximum number of learners to use when fitting the ensemble. If a perfect fit is 
            achieved before reaching this number, the predict method should stop early.
        learning_rate: float
            The weight applied to each weak learner per iteration.
        """
        super().__init__()
        self.weak_learner = weak_learner
        self.num_learners = num_learners
        self.learning_rate = learning_rate
    
    def fit(self, X, y):
        """
        Fit the AdaBoost classifier.

        Parameters
        ----------
        X: numpy.ndarray
            The training data of size (n_samples, n_features).
        y: numpy.ndarray
            The labels of size (n_samples,).
        """
        pass
    
    def predict(self, X):
        """
        Predict the label of each sample in X. Note this is only for binary classification.
        
        Parameters
        ----------
        X: numpy.ndarray
            The training data of size (n_samples, n_features).
        """
        pass



# Example usage
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize and train tree
# tree = DecisionTree(criterion='gini', max_depth=3)
# #tree.predict(X_test)  # should raise NotFittedError
# tree.fit(X_train, y_train)

# # Predict and evaluate
# preds = tree.predict(X_test)
# accuracy = np.mean(preds == y_test)
# print(f"Test accuracy custom: {accuracy:.2f}")

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# # Initialize and train tree
# tree = DecisionTreeClassifier(criterion='gini', max_depth=3)
# tree.fit(X_train, y_train)

# # Predict and evaluate
# preds = tree.predict(X_test)

# accuracy = accuracy_score(y_test, preds)

# print(f"Test accuracy sklearn: {accuracy:.2f}")

# Initialize and train random forest
forest = RandomForest(classifier=DecisionTree(criterion='gini', max_depth=3), num_trees=10, min_features=2)
forest.fit(X_train, y_train)

# Predict and evaluate
preds = forest.predict(X_test)
accuracy = np.mean(preds == y_test)
print(f"Test accuracy custom: {accuracy:.2f}")

from sklearn.ensemble import RandomForestClassifier

# Initialize and train random forest
forest = RandomForestClassifier(n_estimators=10, max_features=2)
forest.fit(X_train, y_train)

# Predict and evaluate
preds = forest.predict(X_test)
accuracy = np.mean(preds == y_test)
print(f"Test accuracy sklearn: {accuracy:.2f}")




