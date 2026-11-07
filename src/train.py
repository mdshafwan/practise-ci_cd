import os
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier, train
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import accuracy_score,f1_score,recall_score
import optuna

os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

def train():
    mlflow.set_tracking_uri('file:./ml_runs')
    mlflow.set_experiment('experiment')

    data_path = "data/customer_churn_dataset-testing-master.csv"
    df = pd.read_csv(data_path)

    x = df.drop('Churn',axis=1)
    y = df['Churn']

    num = x.select_dtypes(include=['int64','float64']).columns.tolist() 
    cat = x.select_dtypes(include=['str']).columns.tolist()

    num_pipe = Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    cat_pipe = Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('scale',OneHotEncoder(handle_unknown='ignore',drop='first'))])
    preprocess = ColumnTransformer([('num',num_pipe,num),('cat',cat_pipe,cat)])

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    def objective(trials):
        with mlflow.start_run(nested=True):
            params={
                'learning_rate':trials.suggest_float('learning_rate',0.1,0.5),
                'max_depth':trials.suggest_int('max_depth',6,8),
                'n_estimators':trials.suggest_int('n_estimators',100,300)
            }

            model = Pipeline([('preprocess',preprocess),('model',XGBClassifier(**params))])
            model.fit(x_train,y_train)
            pred = model.predict(x_test)
            acc = accuracy_score(pred,y_test)
            f1 = f1_score(pred,y_test)
            recall = recall_score(pred,y_test)
            mlflow.log_param('dataset',"churn")
            mlflow.log_params(params)
            mlflow.log_metrics({"acc":acc,"f1":f1,"recall":recall})
            return acc

    with mlflow.start_run(run_name='optuna_name'):
        study = optuna.create_study(direction='maximize')
        study.optimize(objective,n_trials=10)

        best_model = Pipeline([('preprocess',preprocess),('model',XGBClassifier(**study.best_params))])
        best_model.fit(x_train,y_train)

        pred = best_model.predict(x_test)
        acc = accuracy_score(pred,y_test)
        f1 = f1_score(pred,y_test)
        recall = recall_score(pred,y_test)
        mlflow.log_param('dataset',"churn")
        mlflow.log_metrics({"acc":acc,"f1":f1,"recall":recall})
        mlflow.log_params(study.best_params)

if __name__ == "__main__":
    train()
