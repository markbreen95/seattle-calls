from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

class ReturnCluster(BaseEstimator, TransformerMixin):
    def __init__(self, clusters=5): 
        self.clusters = clusters
           
    def fit(self, X):
        self.X=X
        self.model = KMeans(n_clusters = self.clusters, n_init=10, max_iter=300, random_state=42)
        self.model.fit(self.X)
        return self
       
    def transform(self, X):
        self.X=X
        X_ = X.copy() # avoiding modification of the original df
        return self.model.predict(X_)