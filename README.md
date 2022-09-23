# seattle-calls
Repository containing predictive modelling code for Seattle 911 calls

# Requirements 
Found in `requirements.txt`

# Running application
For training, run `train.py`. Note that this requires BigQuery authentication, so the train file will not run unless configured with a BigQuery project that you have access to. 

For inference, run `inference.py`. This is set up such that three different options can be used:

- Querying inference data from BigQuery (unavailable unless you configure it)
- Manually inputting dataframe in `inference.py`
- Loading a csv file from `data/input/`

The inference pipeline will be run for the input dataframe and results will be saved in `data/output/`. The filename will contain a timestamp.

## Data
All data is taken from the [Seattle Real Time Fire 911 Calls](https://data.seattle.gov/Public-Safety/Seattle-Real-Time-Fire-911-Calls/kzjm-xkqj) website.

The data was exported as a .csv file, and then imported into a Google Cloud BigQuery instance. The schema of which is described in the table below:

| Field name      | Type   |
|-----------------|--------|
| Address         | STRING |
| Type            | STRING |
| Datetime        | STRING |
| Latitude        | FLOAT  |
| Longitude       | FLOAT  |
| Report_Location | STRING |
| Incident_Number | STRING |

It was noticed that the `Datetime` field had a strange format, so this was processed and split into two separate fields, `report_date` and `report_time`. Thus, the ultimate schema after this ETL process was:

| Field name      | Type   |
|-----------------|--------|
| Address         | STRING |
| Type            | STRING |
| Latitude        | FLOAT  |
| Longitude       | FLOAT  |
| Incident_Number | STRING |
| report_date     | DATE   |
| report_time     | STRING |

As mentioned, Google BigQuery was used to store and query the data.

## Methods

I wanted to capture some notion of geography within the model. To do this, I first attempted to use an API to query the post code using the latitude and longitude. This, however, would have taken too long. For the full dataset, it would have taken approximately 4155 minutes. 

Instead, I decided to use K-Means clustering on the latitude and longitude to create geographical clusters. The full breakdown of findings here can be found in the supplementary notebook, however, 