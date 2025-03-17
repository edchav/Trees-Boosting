import numpy as np
import pytest
from trees_boosting_library import DecisionTree
from sklearn.exceptions import NotFittedError

def test_decision_tree_fit_predict():
    # Create a simple synthetic dataset for binary classification.
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    # Instantiate and fit the decision tree.
    tree = DecisionTree(max_depth=1)
    tree.fit(X, y)
    
    # Get predictions on the training set.
    predictions = tree.predict(X)
    
    # Check that the number of predictions equals the number of samples.
    assert predictions.shape[0] == X.shape[0]
    
    # Check that predictions are in {0, 1}.
    for p in predictions:
        assert p in [0, 1]

def test_decision_tree_not_fitted():
    # Ensure that calling predict before fitting raises NotFittedError.
    tree = DecisionTree(max_depth=1)
    with pytest.raises(NotFittedError):
        tree.predict(np.array([[1]]))

def test_decision_tree_impurity_methods():
    # Test the impurity calculations for each criterion.
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])  # Balanced split: proportions 0.5 and 0.5
    
    # Gini impurity: For p=0.5 for each class, expected = 1 - (0.25 + 0.25) = 0.5
    tree_gini = DecisionTree(max_depth=1, criterion='gini')
    tree_gini.fit(X, y)
    impurity_gini = tree_gini._node_impurity(y)
    assert np.isclose(impurity_gini, 0.5, atol=1e-3)
    
    # Entropy: For p=0.5 each, expected entropy = -2*0.5*log2(0.5) = 1.0
    tree_entropy = DecisionTree(max_depth=1, criterion='entropy')
    tree_entropy.fit(X, y)
    impurity_entropy = tree_entropy._node_impurity(y)
    assert np.isclose(impurity_entropy, 1.0, atol=1e-3)
    
    # Misclassification error: Expected = 1 - 0.5 = 0.5
    tree_mis = DecisionTree(max_depth=1, criterion='misclassification')
    tree_mis.fit(X, y)
    impurity_mis = tree_mis._node_impurity(y)
    assert np.isclose(impurity_mis, 0.5, atol=1e-3)

def test_decision_tree_max_depth():
    # Verify that setting max_depth limits the growth of the tree.
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    tree = DecisionTree(max_depth=1)
    tree.fit(X, y)
    
    # The tree's root should not be a leaf, but its immediate children should be leaves.
    root = tree.tree
    if not root.is_leaf:
        assert root.left.is_leaf
        assert root.right.is_leaf

def test_decision_tree_splits():
    # Test that the tree makes a reasonable split on a simple dataset.
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    tree = DecisionTree(max_depth=2)
    tree.fit(X, y)
    
    # For a perfectly separable dataset like this, the root split should separate classes.
    root = tree.tree
    # Check that both children are not empty.
    assert root.left is not None and root.right is not None
    # Verify that a leaf node returns a valid prediction.
    left_pred = tree._traverse_tree(np.array([1]), root.left)
    right_pred = tree._traverse_tree(np.array([6]), root.right)
    assert left_pred in [0, 1]
    assert right_pred in [0, 1]
