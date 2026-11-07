import os

def test_data():
    assert os.path.exists('customer_churn_dataset-testing-master.csv')

def test_model():
    assert os.path.exists('best_model.pkl')

def test_set():
    assert os.path.exists('set.pkl')
