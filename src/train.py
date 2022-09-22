from functions import ReturnCluster
from data_loader import DataProcessor
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import numpy as np
import joblib
import time

def prepare_data(df_, clusters, include_hour=False):
  df = df_.copy()
  df.loc[:, 'Cluster'] = clusters

  df.drop(['Longitude'], axis=1, inplace=True)
  df.rename({'Latitude': 'Incident_Number'}, axis=1, inplace=True)

  if include_hour:
    df_agg = df.groupby(['Cluster', 'year', 'month', 'day', 'hour']).count().reset_index()
    X = df_agg[['Cluster', 'month', 'year', 'day', 'hour']]
    y = df_agg[['Incident_Number']].values.reshape((-1,))
  else:
    df_agg = df.groupby(['Cluster', 'year', 'month', 'day']).count().reset_index()
    X = df_agg[['Cluster', 'month', 'year', 'day']]
    y = df_agg[['Incident_Number']].values.reshape((-1,))

  return X, y

def main():
    dproc = DataProcessor(
        credentials_path='niologic-assessment-33f145533e28.json',
        project_id='niologic-assessment'
    )

    df = dproc.query_db('train')

    df_train = df[df['year'] != 2021]
    df_test = df[df['year'] == 2021]

    cluster_pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('cluster', ReturnCluster(clusters=6))
    ])

    train_clusters = cluster_pipeline.fit_transform(df_train[['Longitude', 'Latitude']])
    test_clusters = cluster_pipeline.transform(df_test[['Longitude', 'Latitude']])

    joblib.dump(cluster_pipeline, 'pickled_objects/cluster_pipeline.pkl')

    X_train, y_train = prepare_data(df_train, train_clusters, False)
    X_test, y_test = prepare_data(df_test, test_clusters, False)

    start = time.time()
    reg = HistGradientBoostingRegressor()
    reg.fit(X_train, y_train)
    joblib.dump(reg, 'pickled_objects/regressor.pkl')
    print('Training R2: {}'.format(reg.score(X_train, y_train)))
    print('Test R2: {}'.format(reg.score(X_test, y_test)))
    print('Fit time for HistGradientBoostingRegressor: {}'.format(time.time()-start))
    return True

if __name__ == '__main__':
    main()