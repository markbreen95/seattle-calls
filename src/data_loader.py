from google.cloud import bigquery
from google.oauth2 import service_account

class DataProcessor():
    """
    A class to facilitate execution of BigQuery queries.

    ...

    Attributes
    ----------
    credentials_path : str
        Path to google cloud credentials
    project_id : str
        Google cloud project id

    """

    def __init__(self, credentials_path, project_id):
        """
        Constructs all the necessary attributes for the model pipeline object.

        Parameters
        ----------
            credentials_path : str
                Path to google cloud credentials
            project_id : str
                Google cloud project id
        """
        self.credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        self.project_id = project_id
        self.client = bigquery.Client(credentials=self.credentials, project=self.project_id)

    def query_db(self, mode):
        """
        Performs inference on a dataframe.

        Parameters
        ----------
        mode : str
            Indicates what file to retrieve query from. Should be one of [train, test, inference, validation]

        Returns
        -------
        pd.DataFrame
            Dataframe containing query results
        """
        if mode == 'train':
            with open('queries/train.sql', 'r') as f:
                query_str = f.read()
        elif mode == 'test':
            with open('queries/test.sql', 'r') as f:
                query_str = f.read()
        elif mode == 'inference':
            with open('queries/inference.sql', 'r') as f:
                query_str = f.read()
        else:
            with open('queries/validation.sql', 'r') as f:
                query_str = f.read()

        query_job = self.client.query(query_str)
        return query_job.to_dataframe()