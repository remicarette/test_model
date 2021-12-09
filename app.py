import joblib
import requests
from io import BytesIO
import pickle
from sklearn.linear_model import LinearRegression

model = LinearRegression()

X = [
  [1],
  [2],
  [3]
]

y = [120, 260, 380]

model.fit(X, y)

joblib.dump(model, 'linear_model_2.joblib')
y_predict = model.predict([[10]])

print(y_predict[0])
# github_url = "https://github.com/remicarette/test_model/blob/main/model.joblib?raw=true"

# mfile = BytesIO(requests.get(github_url).content)
# model = pickle.load(mfile)

# print(model)

