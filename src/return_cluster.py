from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor


class ReturnCluster(BaseEstimator, TransformerMixin):
    """
    A class to allow clusters to be returned in the sklearn pipeline.

    ...

    Attributes
    ----------
    clusters : int
        number of clusters to use for the KMeans algorithm

    """

    def __init__(self, n_clusters=5): 
        """
        Constructs all the necessary attributes for the cluster return object.

        Parameters
        ----------
            n_clusters : int
                number of clusters to use for the KMeans algorithm
        """
        self.n_clusters = n_clusters
           
    def fit(self, X):
        """
        Fits the KMeans clustering algorithm to supplied array.

        Parameters
        ----------
        X : np.array
            The numpy array containing latitude and longitude coordinates

        Returns
        -------
        None
        """
        self.X=X
        self.model = KMeans(n_clusters = self.n_clusters, n_init=10, max_iter=300, random_state=42)
        self.model.fit(self.X)
        return self
       
    def transform(self, X):
        """
        Transforms a numpy array of coordinates to return k means clusters.

        Parameters
        ----------
        X : np.array
            The numpy array containing latitude and longitude coordinates

        Returns
        -------
        np.array
            The numpy array containing clusters for input coordinates
        """
        self.X=X
        X_ = X.copy() 
        return self.model.predict(X_)
