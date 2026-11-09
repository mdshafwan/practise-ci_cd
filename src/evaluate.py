import joblib
from sklearn.metrics import accuracy_score

model = joblib.load('model.pkl')
x_test,y_test = joblib.load('set.pkl')
pred = model.predict(x_test)
print(accuracy_score(y_test,pred))