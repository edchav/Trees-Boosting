# implement DecisionTree with a fit and predict method
# Set up the skeleton for the DecisionTree class

import numpy as np

class DecisionTree:
    def __init__(self, criterion='misclassification', max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Base classifier for a decision tree, that will be useed for random forest and boosting.

        Parameters
        ----------
        criterion: str
            Either 'misclassification rate', 'gini impurity', or 'entropy'.
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
        best_impurity = float('inf')
        best_feature_idx = None
        best_threshold = None



    def predict(self, X):
        """
        Predict the label of each sample in X. Note this is only for binary classification.
        
        Parameters
        ----------
        X: numpy.ndarray
            The training data of size (n_samples, n_features)."""
        pass 

    def _node_impurity(self, y):
        """
        Compute the impurity of a node. 

        Parameters
        ----------

        y: numpy.ndarray
            The labels of the node.
        """

        if self.criterion == 'misclassification':
            # 1/D * summation (y != y_not) = 1 - 
            miss_rate = 1 - np.max(np.bincount(y) / y.size)
            return miss_rate
        
        elif self.criterion == 'gini':
            gini = 1 - np.sum((np.bincount(y) / y.size) ** 2)
            return gini
        
        elif self.criterion == 'entropy':
            p = np.bincount(y) / y.size
            entropy = -np.sum(p * np.log2(p))
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
class RandomForest(DecisionTree):
    def __init__(self, num_trees, min_features):
        """
        A random forest classifier that uses DecisionTree as the base classifier.
        
        Parameters
        ----------
        num_trees: int
            The number of trees in the forest.
        min_features: int
            The minimum number of features to consider when looking
            for the best split.
        """
        super().__init__()
        self.num_trees = num_trees
        self.min_features = min_features     
    
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



