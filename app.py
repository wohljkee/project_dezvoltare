from flask import Flask, render_template, request
import pickle
import predict
from predict import parse_pronto, decode_state, predict

with open('data.out', 'rb') as f:
    tfidf_vect = pickle.load(f)

with open('sparse_dict.out', 'rb') as f:
    dict_vectorize = pickle.load(f)

with open('decisionTree.out', 'rb') as f:
    clf = pickle.load(f)

with open('labels.out', 'rb') as f:
    label_encoder = pickle.load(f)

with open('label_encode.out', 'rb') as f:
    label_encoderr = pickle.load(f)



app = Flask(__name__)
entries = []

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get input data from form
        title = request.form.get("Title")
        description = request.form.get("Description")
        build = request.form.get("Build")
        feature = request.form.get("Feature")
        
        data_matrix = parse_pronto(title, description, feature, build)

        prediction = predict(data_matrix)

        decoded_prediction = decode_state(prediction)
    
        return render_template("result.html", prediction=decoded_prediction)
    
    return render_template("form.html")

if __name__ == '__main__':
    app.run(debug=True)