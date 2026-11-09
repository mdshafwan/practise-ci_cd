import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
import joblib

os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
mlflow.set_tracking_uri('file:./mlrun')
mlflow.set_experiment('practise')

df = pd.read_csv('data/customer_churn_dataset-testing-master.csv')

x = df.drop('Churn',axis=1)
y = df['Churn']

num = x.select_dtypes(include=['int64','float64']).columns.tolist() 
cat = x.select_dtypes(include=['str','object']).columns.tolist()

num_pipe = Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
cat_pipe = Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',drop='first'))])
preprocess = ColumnTransformer([('num',num_pipe,num),('cat',cat_pipe,cat)])
model =  Pipeline([('preprocess',preprocess),('model',XGBClassifier(learning_rate=0.1,n_estimators=200,max_depth=6))])
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
pred = model.predict(x_test)
acc = accuracy_score(y_test,pred)
f1 = f1_score(y_test,pred)
recall = recall_score(y_test,pred)
precision = precision_score(y_test,pred)
mlflow.log_params({"learning_rate":0.1,"n_estimators":200,"max_depth":6})
mlflow.log_metrics({"accuracy":acc,"f1":f1,"recall":recall,"precision":precision})
joblib.dump(model,'model.pkl')
joblib.dump((x_test,y_test),'set.pkl')
mlflow.sklearn.log_model(model,artifact_path='model.pkl',serialization_format='pickle')