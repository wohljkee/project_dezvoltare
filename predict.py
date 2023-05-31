import pickle
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from pickle_vect import deserialize_DecisionTree, tfidf_Deserialize, deserialize_dict_vect, deserialize_encode_state
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import word_tokenize
import scipy.sparse

from tokenize_json import tfidf_vectorize

def parse_pronto(title,description,feature,build):

    text_vectorizer = tfidf_Deserialize()
    porter = nltk.PorterStemmer()
    twd = TreebankWordDetokenizer()
    tokens = word_tokenize(title + ' ' + description)
    stemmed_tokens = [porter.stem(t) for t in tokens]
    text = TreebankWordDetokenizer.detokenize(twd, stemmed_tokens)
    text_to_matrix = text_vectorizer.transform([text])


    feature_vectorize = deserialize_dict_vect()
    print(feature_vectorize)
    feature_dict = {'build' : build , 'feature' : feature}
    print(feature_dict)
    feature_matrix = feature_vectorize.transform(feature_dict)
    print("Printing the feature_matrix ", feature_matrix)

    data_matrix = scipy.sparse.hstack((feature_matrix, text_to_matrix))
    print("Printing the data_matrix",data_matrix)
    return data_matrix


# def decode_state(label):
#     encoding = LabelEncoder()
#     encoding.fit(deserialize_encode_state())
#     label_encoder = encoding.inverse_transform(label)
#     print("Printing the label_encoder", label_encoder)
#     return label_encoder[0]

def decode_state(label):
    with open('label_encode.out', 'rb') as f:
        encoding = pickle.load(f)
    
    label_decoder = encoding.inverse_transform(label)
    print("Printing the label_decoder:", label_decoder)
    return label_decoder[0]

def predict(data_matrix):
    clf = deserialize_DecisionTree()

    label_result = clf.predict(data_matrix)
    print("Printing the label_result", label_result)
    return decode_state(label_result)   # un numar - trebuie decodat cu label encoder , decode()

if __name__ == "__main__":

    title = "[CCDT][TDD][AIRSCALE][ASIB+ABIC][eCPRI] RF autonomous reset after RF Lock/Unlock commands from Webem"
    description = "*** DEFAULT TEMPLATE for 2G-3G-4G-5G-SRAN-FDD-TDD-DCM-Micro-Controller common template v1.4.0 (02.07.2021) – PLEASE FILL IT IN BEFORE CREATING A PR AND DO NOT CHANGE /"
    feature = "LTE5111-A\nCB005985-SR-5GC002056-5GC001524"
    build = "SBTS21A_ENB_0000_001999_000009"

    parse = parse_pronto(title,description,feature,build)
    
    prediction = predict(parse)

    print("Prediction: " , prediction)

