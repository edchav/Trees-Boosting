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
        criterion: str {'misclassification', 'gini', 'entropy'}, default = 'misclassification'
            A measurement of the quality of the split. 
        max_depth: int, default = None
            The maximum depth the tree should grow, if none grows until all leaves are pure.
        min_samples_split: int, default = 2
            The minimum number of samples required to split.
        min_samples_leaf: int, default = 1
            The minimum number of samples required for a leaf node.
        """    
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.n_classes = None
        self.n_features = None
        
    def fit(self, X, y):
        """
        Fit the decision tree classifier.

        Parameters
        ----------
        X: numpy.ndarray
            The training data of size (n_samples, n_features).
        y: numpy.ndarray
            The labels of size (n_samples,).

        Returns:
        -------
        self: DecisionTree
            Fitted Model.
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

        # Metadata for node
        node_impurity = self._node_impurity(y)
        class_counts = np.bincount(y, minlength=self.n_classes)

        # create node to hold metadata
        node = Node(
            is_leaf = False,
            feature_idx = None,
            threshold = None,
            left = None,
            right = None,
            prediction = None,
            n_samples = n_samples,
            impurity = node_impurity,
            value = class_counts
        )

        # Stopping conditions
        if(self.max_depth is not None and depth >= self.max_depth or
                n_samples < self.min_samples_split or 
                n_labels == 1):

            node.is_leaf = True
            node.prediction = np.argmax(np.bincount(y))
            return node

        # Find the best split
        feature_idx, threshold = self._best_split(X, y)

        # No split found, create a leaf node
        if feature_idx is None:
            node.is_leaf = True
            node.prediction = np.argmax(np.bincount(y))
            return node

        # Split the data
        feature_values = X[:, feature_idx]
        left_idx = np.where(feature_values <= threshold)
        right_idx = np.where(feature_values > threshold)

        # Check min_samples_leaf constraint
        if len(left_idx[0]) < self.min_samples_leaf or len(right_idx[0]) < self.min_samples_leaf:
            node.is_leaf = True
            node.prediction = np.argmax(np.bincount(y))
            return node

        # Recursively grow left and right subtree
        node.feature_idx = feature_idx
        node.threshold = threshold
        node.left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        node.right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        return node
    
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
        n_samples = X.shape[0]
        if n_samples <= 1:
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
                left_weight = len(left_idx[0]) / n_samples
                right_weight = len(right_idx[0]) / n_samples

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
            The training data of size (n_samples, n_features).
        
        Returns
        -------
        numpy.ndarray
            The predicted class labels.
        """

        if self.tree is None:
            raise NotFittedError('Estimator not fitted, call `fit` first.')
        
        predictions = []
        for x in X:
            predictions.append(self._traverse_tree(x, self.tree))
        return np.array(predictions)

    def _traverse_tree(self, x, node):
        """
        Traverse the tree to find the prediction for a given sample.
        
        Parameters
        ----------
        x: numpy.ndarray
            The sample of size (n_features,).
        node: Node
            The current node being evaluated.
        
        Returns
        -------
        int
            The predicted class label.
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
        
        Returns
        -------
        float
            The impurity of the node. 
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
     
    def export_graphviz(self, out_file, feature_names=None, class_names=None, 
                       max_depth=None, rounded=False, filled=False):
        """
        Export the decision tree in DOT format.

        Parameters
        ----------
        out_file: file object
            Open file to write DOT file.
        feature_names: list of str, (optional)
            Names of the features.
        class_names: list of str, (optional)
            Names of the classes.
        max_depth: int, (optional)
            Max depth to export
        rounded: bool, (optional)
            If True rounds corners for the nodes.
        filled: bool, (optional)
            If True, colors in nodes. 
        """
        # Start the DOT file
        out_file.write("digraph Tree {\n")
        out_file.write('node [shape=box, fontname="helvetica"];\n')
        
        # Counter for assigning unique IDs to nodes
        node_id_counter = [0]
        
        def recurse(node, depth):
            """
            Recursively build the tree visualization
            
            Parameters
            ----------
            node: Node
                Current node to process.
            depth: int 
                Current depth in the tree.
            
            Returns
            -------
            int
                Unique ID assigned to the current node.
            """
            # Assign a unique ID to this node
            current_id = node_id_counter[0]
            node_id_counter[0] += 1

            # Check if we've reached max depth and should truncate
            if max_depth is not None and depth >= max_depth:
                out_file.write(f'{current_id} [label="...", shape=box, color="lightgrey"];\n')
                return current_id

            # Get label for the node
            label_lines =  []
            
            # If internal node, recursively process children
            if not node.is_leaf:
                if feature_names is not None and node.feature_idx is not None:
                    feature_name = feature_names[node.feature_idx]
                else:
                    feature_name = f'X[{node.feature_idx}]'
                label_lines.append(f'{feature_name} <= {node.threshold:.2f}')
            
            # Add metadata to the node
            if node.impurity is not None:
                label_lines.append(f'impurity = {node.impurity:.2f}')
            if node.n_samples is not None:
                label_lines.append(f'samples = {node.n_samples}')
            if node.value is not None:
                label_lines.append(f'value = {node.value}')
            
            # Leaf node, display class label
            if node.is_leaf:
                if class_names is not None:
                    class_name = class_names[node.prediction]
                else:
                    class_name = node.prediction
                label_lines.append(f'class = {class_name}')
            

            label = '\\n'.join(label_lines)

            # Style attributes
            style_list = []
            if filled:
                style_list.append('filled')
            if rounded:
                style_list.append('rounded')
            style_attr = f'style="{", ".join(style_list)}"'

            if filled:
                if node.is_leaf:
                    color = 'lightblue'
                else:
                    color = 'orange'
            else:
                color = 'black'

           # Write the current node definition.
            out_file.write(f'{current_id} [label="{label}", shape=box, {style_attr}, color="{color}"];\n')
        
            # If not a leaf, recursively write child nodes and edge definitions.
            if not node.is_leaf:
                left_id = recurse(node.left, depth+1)
                right_id = recurse(node.right, depth+1)
                out_file.write(f'{current_id} -> {left_id} [label="yes"];\n')
                out_file.write(f'{current_id} -> {right_id} [label="no"];\n')
            
            return current_id

        # Begin recursion from the root.
        recurse(self.tree, 0)
        out_file.write("}\n")
        
class Node:
    def __init__(self, is_leaf = False, feature_idx=None, threshold=None, 
                 left=None, right=None, prediction=None, n_samples = None,
                impurity = None, value = None):
        """
        Node in decision tree
        
        Parameters
        ----------
        is_leaf: bool
            True if the node is a leaf
        feature_idx: int, (optional)
            Index of feature to split on
        threshold: float, (optional)
            Value to split feature on
        left: Node, (optional)
            Left child node
        right: Node, (optional)
            Right child node
        Prediction: int or float (optional)
            Value at leaf node (class label)
        n_samples: int, (optional)
            Number of samples in the node
        impurity: float, (optional)
            Impurity of the node
        value: numpy.ndarray, (optional)
            Class distribution of samples in the node
        """
        self.is_leaf = is_leaf
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction 
        self.n_samples = n_samples
        self.impurity = impurity
        self.value = value

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
        
        Returns
        -------
        self: RandomForest
            Fitted Model.
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

class AdaBoost():
    def __init__(self, weak_learner, num_learners, learning_rate):
        """
        An AdaBoost classifier that uses DecisionTree as the weak learners.
        
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
        self.alphas = [] # list of learner weights
        self.learners = []
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit the AdaBoost classifier.

        Parameters
        ----------
        X: numpy.ndarray
            The training data of size (n_samples, n_features).
        y: numpy.ndarray
            The labels of size (n_samples,).
        
        Returns
        -------
        self: AdaBoost
            Fitted Model.
        """
        n_samples = X.shape[0]

        # Transform labels to -1 and 1
        y_transformed = np.copy(y)
        unique_labels = np.unique(y)
        if np.array_equal(unique_labels, np.array([0, 1])):
            y_transformed = 2 * y_transformed - 1
        elif np.array_equal(unique_labels, np.array([-1, 1])):
            pass
        else:
            raise ValueError('Labels must be in {0, 1} or {-1, 1}')
    
        # Initialize weights
        weights = np.ones(n_samples) / n_samples

        self.learners = [] 
        self.alphas = []

        for _ in range(self.num_learners):
            # create a copy of the weak learner
            learner = copy.deepcopy(self.weak_learner)

            # normalize weights to probabilities
            probabilties = weights / np.sum(weights)

            # sample with replacement according to weights
            sample_indices = np.random.choice(n_samples, size = n_samples, replace = True, p = probabilties)
            X_sample = X[sample_indices]
            y_sample = y_transformed[sample_indices]

            # convert to 0, 1 labels for weak learner (DecisionTree)
            y_sample = (y_sample + 1) // 2

            # fit the weak learner
            learner.fit(X_sample, y_sample)

            # make predictions and convert to -1, 1 labels
            predictions = learner.predict(X)
            predictions = 2 * predictions - 1

            # calculate weighted error
            misclassified = predictions != y_transformed
            weighted_error = np.sum(weights * misclassified) / np.sum(weights)

            # if error is >= 0.5 then break early (weak learner is worse than random guessing)
            if weighted_error >= 0.5:
                break
            if weighted_error < 1e-10:
                ##alpha = self.learning_rate * 0.5 np.log((1 - 1e-10) / 1e-10)
                alpha = self.learning_rate * np.log((1 - 1e-10) / 1e-10)

                self.alphas.append(alpha)
                self.learners.append(learner)
                break

            # calculate alpha
            ##alpha = self.learning_rate * 0.5 * np.log((1 - weighted_error) / weighted_error)
            alpha = self.learning_rate * np.log((1 - weighted_error) / weighted_error)

            # update weights
            ## weights *= np.exp(-alpha * y_transformed * predictions)
            weights *= np.exp(alpha * (y_transformed != predictions))

            # normalize weights
            weights /= np.sum(weights)

            # store alpha and learner
            self.alphas.append(alpha)
            self.learners.append(learner)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict the label of each sample in X. Note this is only for binary classification.
        
        Parameters
        ----------
        X: numpy.ndarray
            The training data of size (n_samples, n_features).
        
        Returns
        -------
        numpy.ndarray
            The predicted class labels.
        """
        if not self.is_fitted:
            raise NotFittedError('Estimator not fitted, call `fit` first.')
        
        prediction_scores = np.zeros(X.shape[0])
        for alpha, learner in zip(self.alphas, self.learners):
            predictions = learner.predict(X)
            predictions = 2 * predictions - 1
            prediction_scores += alpha * predictions

        return np.sign(prediction_scores)
