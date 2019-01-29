from sklearn.externals import joblib
import flask 
import numpy as np

app = flask.Flask(__name__)
model = None

def load_model():
    '''
    学習済みモデルを読み込む関数
    パスに指定した.pklファイルをmodelに読み込む
    '''

    global model
    print("学習済みモデルを読み込んでいます...")
    model = joblib.load("./model/sample-model.pkl")
    print("ロードが完了しました")

@app.route("/predict",methods=["POST"])
def predict():
    response = {
        "success":False,
        "Content-Type":"application/json"
    }
    if flask.request.method == "POST":
        if flask.request.get_json().get("feature"):
            feature = flask.request.get_json().get("feature")

            feature = np.array(feature).reshape((1,-1))

            response["prediction"] = model.predict(feature).tolist()

            response["success"] = True
    return flask.jsonify(response)

if __name__ =="__main__":
    load_model()
    print("* starting server...")
    app.run(host='0.0.0.0',port=5000)