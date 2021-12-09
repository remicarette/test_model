import joblib
import requests
from io import BytesIO
import pickle
from sklearn.linear_model import LinearRegression

model = LinearRegression()


def train_and_export_model():
    X = [
      [1],
      [2],
      [3],
      [10],
      [32]
    ]

    y = [120, 260, 380, 3029, 9032]

    model.fit(X, y)

    y_predict = model.predict([[10]])

    print(y_predict[0])
    filepath = 'linear_model_2.joblib'
    joblib.dump(model, filepath)
    print(f'export to {filepath}')

def import_form_github_and_predict():
    # joblib.load()
    url = "https://github.com/remicarette/test_model/blob/main/linear_model_2.joblib?raw=true"
    mfile = BytesIO(requests.get(url).content)
    model = joblib.load(mfile)
    print(model)
    y_predict = model.predict([[10]])

    print(y_predict[0])


train_and_export_model()
# import_form_github_and_predict()