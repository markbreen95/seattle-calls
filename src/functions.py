from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
import joblib


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


class ModelPipeline():
    def __init__(self, df, include_hour=True):
        self.df = df
        self.include_hour = include_hour

    def fit(self):
        df = self.df

        cluster_pipeline = Pipeline([
            ('scale', StandardScaler()),
            ('cluster', ReturnCluster(clusters=6))
        ])

        clusters = cluster_pipeline.fit_transform(df[['Longitude', 'Latitude']])

        joblib.dump(cluster_pipeline, 'pickled_objects/cluster_pipeline.pkl')

        self.clusters = clusters

        X, y  = self.prepare_data()

        reg = HistGradientBoostingRegressor()
        reg.fit(X, y)

        joblib.dump(reg, 'pickled_objects/regressor.pkl')
    
        print('Training R2: {}'.format(reg.score(X, y)))

    def inference(self):
        df = self.df

        cluster_pipeline = joblib.load('pickled_objects/cluster_pipeline.pkl')

        clusters = cluster_pipeline.transform(df[['Longitude', 'Latitude']])

        self.clusters = clusters

        X, y = self.prepare_data()

        reg = joblib.load('pickled_objects/regressor.pkl')
        
        print('Inference R2: {}'.format(reg.score(X, y)))



    def prepare_data(self):
        df = self.df
        df.loc[:, 'Cluster'] = self.clusters

        df.drop(['Longitude'], axis=1, inplace=True)
        df.rename({'Latitude': 'Incident_Number'}, axis=1, inplace=True)

        if self.include_hour:
            df_agg = df.groupby(['Cluster', 'year', 'month', 'day', 'hour']).count().reset_index()
            X = df_agg[['Cluster', 'month', 'year', 'day', 'hour']]
            y = df_agg[['Incident_Number']].values.reshape((-1,))
        else:
            df_agg = df.groupby(['Cluster', 'year', 'month', 'day']).count().reset_index()
            X = df_agg[['Cluster', 'month', 'year', 'day']]
            y = df_agg[['Incident_Number']].values.reshape((-1,))

        return X, y


