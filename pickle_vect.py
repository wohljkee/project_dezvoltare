from scipy import sparse
import tokenize_json
import pickle, json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from pymongo import MongoClient
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import scipy.sparse

client = MongoClient("localhost", 27017)
with open("prontos.json", "r") as data_file:    
    data = json.load(data_file)

db = client["database_test"]    #  creating a database inside the mongodb client
collection = db["collection_test"]  #then create a new collection in that database


# Serialize and deserialize the feature_extract() function
def serialize_features_extract(database_test, collection_test, file_path = "dict_vectorize.out"):
    db = client[database_test]
    collection = db[collection_test]

    feature_list = []
    for doc in collection.find():
        features = {} 
        features['feature'] = doc.get('feature', '')
        features['build'] = doc.get('build', '')
        feature_list.append(features)

    dict_vectorizer = DictVectorizer()
    sparse_matrix = dict_vectorizer.fit_transform(feature_list)

    #Serialization : 
    with open(file_path, "wb") as file_serialized:
        pickle.dump(dict_vectorizer, file_serialized)

    with open("sparse_dict.out", "wb") as file_serialized:
        pickle.dump(sparse_matrix, file_serialized)

def deserialize_dict_vect(file_path = "dict_vectorize.out"):
    with open(file_path, "rb") as file_deserialized:
        return pickle.load(file_deserialized)
    
def deserialize_sparse_dict(file_path = "sparse_dict.out"):
    with open(file_path, "rb") as file_deserialized:
        return pickle.load(file_deserialized)

# Serialize and deserialize the label_encodeState() 
def label_encodeState(database_test, collection_test, file_path="label_encode.out"):
    db = client[database_test]
    collection = db[collection_test]
    state_list = []
    cursor = collection.find({})
    for doc in cursor:
        state = doc['state'] # setting a variable with 'state' field items
        if(state == "Closed" or state != "Correction Not Needed"):
            state = 'Software issue'
        elif(state == "Correction Not Needed"):
            state = "CNN"
        state_list.append(state)
   
    print(state_list)
    label_encoder = LabelEncoder()

      # Check if the label_encoder has been fitted before
    if hasattr(label_encoder, 'classes_'):
        # Use the existing label_encoder
        label_encoder.classes_ = np.append(label_encoder.classes_, np.unique(state_list))
    else:
        # Fit the label_encoder on the first encountered labels
        label_encoder.fit(state_list)

    encoded_states = label_encoder.transform(state_list)
    print(encoded_states)

    with open(file_path, "wb") as file_serialized:
        pickle.dump(label_encoder, file_serialized)

    with open("labels.out", "wb") as file_serialized:
        pickle.dump(encoded_states, file_serialized)

    return encoded_states

def deserialize_label_encoder(file_path = "label_encode.out"):
    with open(file_path, "rb") as file_deserialized:
        label_encoder = pickle.load(file_deserialized)
    return label_encoder

def deserialize_encode_state(file_path = "labels.out"):
    with open(file_path, "rb") as file_deserialized:
        encoded_states  = pickle.load(file_deserialized)
    return encoded_states


#Serialize and deserialize on TfidfVectorizer() for processed_text
def tfidf_concat_vectorize(database_test, collection_test, file_path = "tfidf_vect.out"):
    db = client[database_test]
    collection = db[collection_test]

    # concatenate the title and description fields and process the text 
    data = []
    for doc in collection.find():
        pr_text = doc["title"] + " " + doc["description"]
        tokens = tokenize_json.tokenize_text(pr_text)
        stemmed_tokens = tokenize_json.stem_tokens(tokens)
        processed_text =' '.join(stemmed_tokens)
        data.append({"_id": str(doc["_id"]), "processed_text" : processed_text})
        collection.update_one({"_id":doc["_id"]}, {"$set":{"processed_text": processed_text}})

    # performing TFIDF Vectorization on the processsed_text

    vectorizer = TfidfVectorizer()
    sparse_matrix = vectorizer.fit_transform(doc["processed_text"] for doc in data)

    with open(file_path, 'wb') as file_serialized:
        pickle.dump(vectorizer, file_serialized)

    with open("data.out", 'wb') as file_serialized:
        pickle.dump(sparse_matrix, file_serialized)
    
    return sparse_matrix, vectorizer
        

def tfidf_Deserialize(file_path = "tfidf_vect.out"):
    with open(file_path, 'rb') as file_deserialized:
       vectorizer = pickle.load(file_deserialized)
    return vectorizer

def tfidf_Data(file_path = "data.out"):
    with open(file_path, 'rb') as file_deserialized:
       sparse_matrix = pickle.load(file_deserialized)
    return sparse_matrix

#Serialize and deserialize the DecisionTreeAlgorithm :
def serialize_train_decision_tree(database_test, collection_test, file_path= "decisionTree.out"):
    db = client[database_test]
    collection = db[collection_test]

    sparse_matrix = tfidf_Data()
    labels = deserialize_encode_state()
    dict_sparse = deserialize_sparse_dict()

    data_matrix = scipy.sparse.hstack((dict_sparse, sparse_matrix))
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data_matrix, labels, test_size=0.2, random_state=42)

    # Replace missing values with mean of each feature
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    #Train a decision tree classifier : 
    clf = DecisionTreeClassifier(random_state=30)
    clf.fit(X_train, y_train)
  
    #Test the model on the testing set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,  y_pred)
    print('Accuracy: {:.2f}%'.format(accuracy * 100))

    with open(file_path,'wb') as file_serialized:
        pickle.dump(clf, file_serialized)

    return clf



def deserialize_DecisionTree(file_path = "decisionTree.out"):
    with open(file_path, "rb") as file_deserialized:
        clf = pickle.load(file_deserialized)
    return clf

if __name__ == "__main__":
    
    database_test = "database_test"
    collection_test = "collection_test"

    # Serialize and deserialize feature_extract() function
    serialize_features_extract(database_test, collection_test)
    deserialize_dict_vect()
    # Serialize and deserialize label_encodeState() function
    label_encodeState(database_test, collection_test)
    deserialize_encode_state()
    deserialize_label_encoder()
    
    # Serialize and deserialize TfidfVectorizer function
    tfidf_concat_vectorize(database_test, collection_test)
    tfidf_Deserialize()

    # tfidf_concat_vectorize(database_test,collection_test)
    serialize_features_extract(database_test,collection_test)

    # Serialize and deserialize DecisionTreeAlgorithm function
    serialize_train_decision_tree(database_test, collection_test)
    deserialize_DecisionTree()


client.close()
