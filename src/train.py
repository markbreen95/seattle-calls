from functions import ReturnCluster
import pandas as pd
import random

df = pd.DataFrame(
    {'col1': [random.random() for i in range(25)],
    'col2': [random.random() for i in range(25)]
    }
)

print(ReturnCluster().fit_transform(df))

print('Hello World!')