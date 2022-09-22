from google.cloud import bigquery
from google.oauth2 import service_account

class DataProcessor():
    def __init__(self, credentials_path, project_id):
        self.credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        self.project_id = project_id
        self.client = bigquery.Client(credentials=self.credentials, project=self.project_id)

    def query_db(self, mode):
        if mode == 'train':
            with open('train.sql', 'r') as f:
                query_str = f.read()
        elif mode == 'test':
            with open('test.sql', 'r') as f:
                query_str = f.read()
        else:
            with open('validation.sql', 'r') as f:
                query_str = f.read()

        query_job = self.client.query(query_str)
        return query_job.to_dataframe()