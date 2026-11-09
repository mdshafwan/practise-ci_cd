import os 
import pandas as pd

def validate():

    df = pd.read_csv('src/customer_churn_dataset-testing-master.csv')
    
    if not os.path.exists('src/customer_churn_dataset-testing-master.csv'):
        print("the dataset was not exist")
        exit(1)

    print(df.info())
    print(df.describe())


if __name__ == "__main__":
    validate()