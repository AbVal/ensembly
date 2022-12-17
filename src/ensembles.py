import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.random_state = None

        self.tree_list = []
        self.feature_subset_list = []

        for param, value in trees_parameters.items():
            self.__setattr__(param, value)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        np.random.seed(self.random_state)

        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3
        elif isinstance(self.feature_subsample_size, float):
            self.feature_subsample_size = int(X.shape[1] * self.feature_subsample_size)

        for _ in range(self.n_estimators):
            bag = np.random.choice(X.shape[0], X.shape[0], replace=True)
            feature = np.random.choice(X.shape[1], self.feature_subsample_size, replace=False)

            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X[bag, feature], y[bag])

            self.tree_list.append(tree)
            self.feature_subset_list.append(feature)

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        tree_count = len(self.tree_list)
        if tree_count == 0:
            raise RuntimeError('Unable to predict: no trees in forest')

        prediction = np.zeros(X.shape[0])

        for tree, feature_subset in zip(self.tree_list, self.feature_subset_list):
            prediction += tree.predict(X[:, feature_subset])

        return prediction / tree_count


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.random_state = None

        self.tree_list = []
        self.alpha_list = []
        self.feature_subset_list = []

        for param, value in trees_parameters.items():
            self.__setattr__(param, value)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        np.random.seed(self.random_state)

        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3
        elif isinstance(self.feature_subsample_size, float):
            self.feature_subsample_size = int(X.shape[1] * self.feature_subsample_size)

        y_pred = np.zeros_like(y)

        for _ in range(self.n_estimators):
            antigrad = 2 * (y - y_pred)
            feature = np.random.choice(X.shape[1], self.feature_subsample_size, replace=False)

            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X[:, feature], antigrad)
            tree_pred = tree.predict(X)

            alpha = minimize_scalar(lambda x, tp=tree_pred: mean_squared_error(y, y_pred + x * tp)).x

            self.tree_list.append(tree)
            self.alpha_list.append(alpha)
            self.feature_subset_list.append(feature)

            y_pred = y_pred + alpha * self.learning_rate * tree_pred

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        prediction = np.zeros(X.shape[0])

        for tree, feature_subset, alpha in zip(self.tree_list, self.feature_subset_list, self.alpha_list):
            prediction += alpha * self.learning_rate * tree.predict(X[:, feature_subset])

        return prediction
