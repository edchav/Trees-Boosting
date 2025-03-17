import numpy as np
import pytest
from trees_boosting_library import RandomForest, DecisionTree
from sklearn.exceptions import NotFittedError

def test_bootstrap_sampling():
    X = np.array([[1,2], [3,4], [5,6], [7,8], [9,10]])
    y = np.array([0,1,0,1,0])
    
    rf = RandomForest(DecisionTree(), num_trees=5, min_features=1)
    rf.fit(X, y)
    
    # Verify all trees have valid feature subsets
    for tree in rf.trees:
        assert 1 <= len(tree.feature_indices) <= 2  # 2 total features
        assert np.all(np.isin(tree.feature_indices, [0, 1]))

def test_feature_selection():
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)

    rf = RandomForest(DecisionTree(), num_trees=10, min_features=3)
    rf.fit(X, y)

    for tree in rf.trees:
        n_selected_features = len(tree.feature_indices)
        assert 3 <= n_selected_features <= 10
        assert len(np.unique(tree.feature_indices)) == n_selected_features

def test_tree_diveristy():
    X = np.random.rand(50, 5)
    y = np.random.randint(0, 2, 50)

    # Use a moderate max_depth to allow for some variability
    rf = RandomForest(DecisionTree(max_depth=2), num_trees=10, min_features=3)
    rf.fit(X, y)

    # Compare first two trees: ensure they used different feature subsets.
    assert not np.array_equal(rf.trees[0].feature_indices, rf.trees[1].feature_indices)

    # Compare last two trees.
    assert not np.array_equal(rf.trees[-1].feature_indices, rf.trees[-2].feature_indices)

    # Check prediction diversity over multiple test samples.
    test_samples = X[:10]
    diverse_found = False
    for i in range(test_samples.shape[0]):
        preds = [tree.predict(test_samples[i:i+1, tree.feature_indices])[0] for tree in rf.trees]
        if len(np.unique(preds)) > 1:
            diverse_found = True
            break
    assert diverse_found

def test_majority_voting():
    class MockTree:
        def __init__(self, preds):
            self.feature_indices = np.array([0])  # Dummy feature indices
            self.preds = preds
        def predict(self, X):
            return self.preds
    
    rf = RandomForest(None, num_trees=3, min_features=1)
    rf.trees = [
        MockTree(np.array([0, 0, 1, 1, 1])),
        MockTree(np.array([0, 0, 1, 1, 1])),
        MockTree(np.array([0, 0, 1, 1, 1]))
    ]
    
    dummy_X = np.zeros((5, 1))  # Matches the expected feature shape.
    expected = np.array([0, 0, 1, 1, 1])
    assert np.array_equal(rf.predict(dummy_X), expected)

def test_edge_cases():
    X = np.array([[0], [1]])  # Shape (2,1)
    y = np.array([0, 1])
    
    rf = RandomForest(DecisionTree(), num_trees=1, min_features=1)
    rf.fit(X, y)
    
    # Verify prediction shape.
    test_X = np.array([[0], [1]])
    preds = rf.predict(test_X)
    assert preds.shape == (2,)
    
    # Verify feature indices: for a single-feature dataset, the only possible index is 0.
    assert len(rf.trees[0].feature_indices) == 1
    assert rf.trees[0].feature_indices[0] == 0

def test_integration():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create a synthetic dataset.
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Initialize and fit the random forest.
    rf = RandomForest(
        DecisionTree(criterion='gini', max_depth=3),
        num_trees=50,
        min_features=3
    )
    rf.fit(X_train, y_train)
    
    # Basic performance check.
    preds = rf.predict(X_test)
    accuracy = np.mean(preds == y_test)
    assert accuracy > 0.7  # Expect reasonable accuracy on synthetic data.
