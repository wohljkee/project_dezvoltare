from flask import Flask, render_template, request
import pickle
import tokenize_json, pickle_vect, predict
from predict import parse_pronto, decode_state, predict
# Load the serialized objects
with open('data.out', 'rb') as f:
    tfidf_vect = pickle.load(f)

with open('sparse_dict.out', 'rb') as f:
    dict_vectorize = pickle.load(f)

with open('decisionTree.out', 'rb') as f:
    clf = pickle.load(f)

with open('labels.out', 'rb') as f:
    label_encoder = pickle.load(f)



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
        
    #     # Make prediction using input data and trained model
    #     input_data = [title, description, build, feature]
    #     prediction = predict.predict_state(input_data, clf, tokenize_json.tfidf_vectorize, tokenize_json.concat_full, label_encoder)
        
    #     # Add input data and prediction to entries list
    #     entries.append({"Title": title, "Description": description, "Build": build, "Feature": feature, "Prediction": prediction})
    
    # return render_template('form.html')
    # Parse the input data
        data_matrix = parse_pronto(title, description, feature, build)

        # Make a prediction
        prediction = predict(data_matrix)

        # Decode the prediction label
        decoded_prediction = decode_state(prediction)

        # Render the template with the prediction result
        return render_template("result.html", prediction=decoded_prediction)
    # Render the form template for GET requests
    return render_template("form.html")

if __name__ == '__main__':
    app.run(debug=True)