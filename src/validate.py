import os 
import pandas as pd

def validate():

    df = pd.read_csv('data/customer_churn_dataset-testing-master.csv')
    
    if not os.path.exists('data/customer_churn_dataset-testing-master.csv'):
        print("the dataset was not exist")
        exit(1)

    print(df.info())
    print(df.describe())


if __name__ == "__main__":
    validate()