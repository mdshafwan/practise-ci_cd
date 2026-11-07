import os
import pandas as pd

data_path = 'data/customer_churn_dataset-testing-master.csv'

def validate():
    if not os.path.exists(data_path):
        print('the dataset was not exist')
        exit(1)

    df = pd.read_csv(data_path)
    print(df.describe())
    print(df.info())

if __name__ == "__main__":
    validate()