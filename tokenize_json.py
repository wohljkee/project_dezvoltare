""" Natural Language Toolkit  - nltk in python"""

""" Simple things to do using NLTK"""
import pprint
import nltk
import json, pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from pymongo import MongoClient
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

#Splitting the training and test data 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score  
from scipy.sparse import hstack
import scipy.sparse
# from nltk.corpus import words
# nltk.download('stopwords')
# nltk.download('punkt')

client = MongoClient("localhost", 27017)

with open("prontos.json", "r") as data_file:    
    data = json.load(data_file)

db = client["database_test"]   
collection = db["collection_test"]

#TASK 1
def desc_tokenize(client:MongoClient):
    db = client["database_test"]
    collection = db["collection_test"]
    elem = collection.find({})
    tokens_list = []   
    for doc in elem:
        if isinstance(doc["description"], (str, bytes)):     
            tokens = word_tokenize(doc["description"])  
            collection.update_one({"_id": doc["_id"]}, {'$set': {"tokenized_description": tokens}})  
            tokens_list.append(tokens) 
            
    return tokens_list  

tokens = desc_tokenize(client)

# TASK 2  : Applying different stemming methods on the tokens .
# Method 1
def stem_tokens():
    ps = PorterStemmer()
    stems_list = [] 
    db = client["database_test"]
    collection = db["collection_test"]
    stops = set(stopwords.words("english")) 
    for doc in collection.find({}):
        description = doc['description']
        words = [tok for tok in description.split() if tok.isalpha() and tok.lower() not in stops]
        stem = [ps.stem(tok) for tok in words] 
        collection.update_one({"_id": doc["_id"]}, {'$set': {'tokenized_description': ' '.join(stem)}})
        stems_list.append(' '.join(stem))
    return stems_list

stemming = stem_tokens()

# # TASK 3  :  Rejoin stemmed tokens and use a TfIdf Vectorizer on the descriptions field.

# def rejoin_stemmed_words(stemmed_words):
#     return [' '.join(words) for words in stemmed_words]

# # stemmed_words = stem_tokens()   # same as stemming called above 
# # joined_words = rejoin_stemmed_words(stemmed_words)
# # print(joined_words)

# # tfidf = TfidfVectorizer(tokenizer=stemming, stop_words='english')    
# # y = tfidf.fit_transform()
# # print(tfidf.get_features_names_out())
# # print(y.shape())


# TASK 4 
""" Applying a Tfidf Vectorizer on the raw text from description field """
data_raw = []
for doc in collection.find({}):
    data_raw.append(doc['description'])
tfidf_vectorizer = TfidfVectorizer()    
tfidf_raw = tfidf_vectorizer.fit_transform(data_raw)
# print(tfidf_raw.toarray())
# Initialize the TfidfVectorizer with the desired settings
# tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vocabulary)
# print(tfidf_df)


# def tfidf_vectorize():
#     db = client["database_test"]
#     collection = db["collection_test"]

#     vectorizer = TfidfVectorizer()
#     # Fit the vectorizer to the list of stemmed descriptions
#     tfidf_matrix = vectorizer.fit_transform(stemming)
#     print(tfidf_matrix.shape)       # the shape for the tfidf document (for the prontos.json)       -- Returns : (2637,5392)
#     print(tfidf_matrix.toarray())
    
# tfidf = tfidf_description()

# feature_names = vectorizer.get_feature_names_out()
# tfidf_score = tfidf_matrix.toarray()[0]

# Print the feature names and their corresponding TF-IDF scores if it is a nonzero score in the description 
# for i in range(len(feature_names)):
#     if tfidf_score[i] > 0:
#         print(f"{feature_names[i]}: {tfidf_score[i]}")

def tfidf_scores(stemming):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(stemming)

    feature_names = vectorizer.get_feature_names_out()
    tfidf_score = tfidf_matrix.toarray()[0]

    tfidf_dict = {}
    for i in range(len(feature_names)):
        if tfidf_score[i] > 0 :
            tfidf_dict[feature_names[i]] = tfidf_score[i]

    return tfidf_dict

# TASK WEEK 4
""""Title and description of a PR should be concatenated and then tokenized, stemmed, etc."""
stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

def tokenize_text(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    return filtered_tokens

def stem_tokens(tokens):
    stemmed_tokens = [porter.stem(token) for token in tokens]
    return stemmed_tokens

def concat_full(database_test, collection_test):
    db = client[database_test]
    collection = db[collection_test]
    data = []
    for pr in collection.find():
        pr_text = pr["title"] + " " + pr["description"]
        tokens = tokenize_text(pr_text)
        stemmed_tokens = stem_tokens(tokens)
        processed_text = ' '.join(stemmed_tokens)
        data.append({"_id": str(pr["_id"]), "processed_text" : processed_text})
        collection.update_one({"_id": pr["_id"]},{"$set": {"processed_text": processed_text}})
    
    dict_vectorizer = DictVectorizer()
    sparse_matrix = dict_vectorizer.fit_transform(data)
    # print(sparse_matrix)
    return sparse_matrix.toarray()      

text_concatenate = concat_full('database_test', 'collection_test')
# print(text_concatenate)


"""TfidfVectorizer() on the processed_text (after tokenizing and stemming)"""
def tfidf_vectorize(database_test, collection_test):
    db = client[database_test]
    collection = db[collection_test]

    stemmed_tokens = []
    for doc in collection.find():
        stemmed_tokens.append(doc["processed_text"])

    text = ' '.join(stemmed_tokens)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text])

    return vectors

vectorss = tfidf_vectorize('database_test', 'collection_test')

"""TASK 2 """
# Turn the other data into useful features for our model (DictVectorizer, OneHotEncoder)
# Getting from the pronto : 'build' , 'feature' fields for now

# Using DictVectorizer

def features_extract(database_test, collection_test):
    db = client[database_test]
    collection = db[collection_test]

    feature_list = []
    for doc in collection.find():
        features = {} 
        features['feature'] = doc.get('feature', '')
        features['build'] = doc.get('build', '')
        feature_list.append(features)
    # print(feature_list)
    dict_vectorizer = DictVectorizer()
    sparse_matrix = dict_vectorizer.fit_transform(feature_list)
    # print(sparse_matrix)
    return sparse_matrix.toarray()

features = features_extract('database_test', 'collection_test')
# print(features)


# Encoding with LabelEncoder for 'state' field ( to have only 2 values ( Closed , Correction Not Needed ) not more than 2 like 'groupInCharge')

def encode_state(database_test, collection_test, field_name):
    db = client[database_test]
    collection = db[collection_test]

    documents = collection.find({})

    field = [doc[field_name] for doc in documents]
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(field)
    df = pd.DataFrame({'state' : field, 'encoded_labels' : encoded_labels})
    return df

encoded_label_state = encode_state('database_test', 'collection_test', 'state')
# print(encoded_label_state)



def splitData(database_test, collection_test, test_size=0.2):
    db = client[database_test]
    collection = db[collection_test]
    X = []
    y = []
    for pr in collection.find():
        pr_text = pr['title'] + " " + pr['description']
        X.append(pr_text)
        y.append(pr["groupInCharge"])

    # Using the CountVectorizer to  create a sparse matrix of word counts 
    vectorizer = CountVectorizer()
    X_sparse = vectorizer.fit_transform(X)  # sparse matrix on the title and description
    # Split the data into training and test sets :
    X_train, X_test, y_train, y_test = train_test_split(X_sparse, y, test_size=test_size, random_state=42)
    # Training using a Decision Tree Classifier:
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    # Evaluate the classifier on the test set 
    score = clf.score(X_test, y_test)
    # print("The accuracy is : ", score)
    # print("Accuracy: {:.2f}%".format(score * 100))
    return clf

splitting_data = splitData('database_test', 'collection_test', 0.2)



"""WEEK 6"""

# Concatenare sparse matrixes la cele 2 functii (concat_full(), feature_extract()) cu HSTACK

def concatenate(database_test, collection_test):
    feature_extract = features_extract(database_test, collection_test)      # (2637,2460)
    concat_text = concat_full(database_test, collection_test)   # this is (2637, 5273)

    # print("Features_extract : ", feature_extract.shape)
    # print("Concat_full : ", concat_text.shape)

    concat_text_reshaped = np.reshape(concat_text, (concat_text.shape[0], concat_text.shape[1]))    
  
    concat_text_sparse = scipy.sparse.csr_matrix(concat_text_reshaped)

    # Concatenate feature_extract and concat_text_sparse horizontally
    full_concatenate = hstack([feature_extract, concat_text_sparse])
    # print(full_concatenate.toarray()) 
    # print(full_concatenate.shape)   # the shape for this concatenate function : (2637,7733)
    return full_concatenate

concatenated_sparse_matrices = concatenate('database_test','collection_test')
# print("Concatenated Sparse Matrices: ", concatenated_sparse_matrices)



# DecisionTreeClassifier() for this concatenate() function which have 2 sparse matrices concatenated 
def train_decision_tree(database_test,collection_test):

    # Concatenate the sparse matrices
    sparse_matrix = concatenate(database_test, collection_test)
    labels = label_encodeState(database_test, collection_test)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(sparse_matrix, labels, test_size=0.2, random_state=42)

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

    return clf


# clf = train_decision_tree('database_test', 'collection_test')

# np.set_printoptions(threshold=np.inf)

# modify this to exclude not_state values but still get a null value for the array
def label_encodeState(database_test,collection_test):
    db = client[database_test]
    collection = db[collection_test]
    state_list = []
    cursor = collection.find({})
    for doc in cursor:
        state = doc['state'] # setting a variable with 'state' field items
        if(state !='Correction Not Needed'):
            state_list.append('Software issue')
        else: 
            state_list.append('CNN')

    count_nonzero = np.count_nonzero(state_list)
    print("Number of non-zero elements in state_list:", count_nonzero)
    print(state_list)
    label_encoder = LabelEncoder()
    encoded_states = label_encoder.fit_transform(state_list)
    
    return encoded_states

encodingState = label_encodeState('database_test','collection_test')

if __name__ == "__main__":
    # gaussian_NB('database_test', 'collection_test')
    train_decision_tree('database_test', 'collection_test')