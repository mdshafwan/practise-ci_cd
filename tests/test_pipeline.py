import os 

def test_data():
    assert os.path.exists("data/customer_churn_dataset-testing-master.csv")

def train_model():
    assert os.path.exists('model.pkl')

def test_data():
    assert os.path.exists('set.pkl')