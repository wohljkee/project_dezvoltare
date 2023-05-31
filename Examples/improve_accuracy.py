""" Natural Language Toolkit  - nltk in python"""

""" Simple things to do using NLTK"""
import pprint
import nltk
import json
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from pymongo import MongoClient
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#Splitting the training and test data 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#hstack for concatenating 2 sparse matrix horizontally
from scipy.sparse import hstack
import scipy.sparse
# from nltk.corpus import words
# nltk.download('stopwords')
# nltk.download('punkt')

client = MongoClient("localhost", 27017)

with open("prontos.json", "r") as data_file:    
    data = json.load(data_file)

db = client["database_test"]    #  creating a database inside the mongodb client
collection = db["collection_test"]  #then create a new collection in that database

""" TASK 1"""

def desc_tokenize(client:MongoClient):
    db = client["database_test"]
    collection = db["collection_test"]
    elem = collection.find({})# variable with all the elements inside the collection
    tokens_list = []    # creating an empty array for the tokenized data 
    for doc in elem:  # for every document inside the collection 
        # print(doc["description"])
        if isinstance(doc["description"], (str, bytes)):        # verify if the 'description' field from the collection is of type "string or bytes"
            tokens = word_tokenize(doc["description"])  # used the word_tokenize() method on the ['description'] field to tokenize every words in that field
            collection.update_one({"_id": doc["_id"]}, {'$set': {"tokenized_description": tokens}})   # here we updated the collection for each string in the 'description' field
            tokens_list.append(tokens)  # here we added everything that was tokenized in the empty array we created
            
    return tokens_list  

tokens = desc_tokenize(client)
# print(tokens)
# We can see the result in the mongosh , using the command : db.collection.find() 
# or to extract one single field we can use the projection : db.collection.find({}
#  {"description": 1}) is a projection object that includes only the description field in the query results.
#  # the value 1 in the projection object specifies that the field should be included in the results

"""TASK 2  :: Apply different stemming methods on the tokens """

"""Method 1"""
def stem_tokens():
    ps = PorterStemmer()
    stems_list = [] # this is a empty list to append all the stemmed data
    db = client["database_test"]
    collection = db["collection_test"]
    stops = set(stopwords.words("english")) # set of english stopwords
    for doc in collection.find({}):
        description = doc['description']
        words = [tok for tok in description.split() if tok.isalpha() and tok.lower() not in stops]
        stem = [ps.stem(tok) for tok in words] 
        collection.update_one({"_id": doc["_id"]}, {'$set': {'tokenized_description': ' '.join(stem)}})
        # stems_list.append(stem) # the correct one to return the stem list of tokens
        stems_list.append(' '.join(stem))
    return stems_list

stemming = stem_tokens()
# The output of tokenized_description after stemming without stopwords: '...'
"""tokenized_description: 'default templat common templat pleas fill creat pr chang lremov section tempat detail test need specif made fault report custom made ticket perform gnb softwar upgrad 
releas rf softwar current lte primari link expect gnb sw activ """


"""SNOWBALL STEMMING using SnowballStemming()"""

def snowball_stemming():
    snowball = SnowballStemmer(language="english")
    stems_list = [] # an empty list to append all the data after stemming
    db = client["database_test"]
    collection = db["collection_test"]
    stops = set(stopwords.words("english"))   # set of english stopwords
    for doc in collection.find({}):
        description = doc['description']
        words = [tok for tok in description.split() if tok.isalpha() and tok.lower() not in stops]
        stem = [snowball.stem(tok) for tok in words]
        collection.update_one({"_id": doc["_id"]},{'$set': {'tokenized_description_snowball': ' '.join(stem)}})
        stems_list.append(' '.join(stem))
    return stems_list
    
snowballStem = snowball_stemming()
#The output is : 

def lancaster_stemmming():
    lancaster = LancasterStemmer()
    stems_list = [] # an empty list to append all the data after stemming
    db = client["database_test"]
    collection = db["collection_test"]
    stops = set(stopwords.words("english"))   # set of english stopwords
    for doc in collection.find({}):
        description = doc['description']
        words = [tok for tok in description.split() if tok.isalpha() and tok.lower() not in stops]
        stem = [lancaster.stem(tok) for tok in words]
        collection.update_one({"_id": doc["_id"]},{'$set': {'tokenized_description_lancaster': ' '.join(stem)}})
        stems_list.append(' '.join(stem))
    return stems_list

lancaster = lancaster_stemmming()
### DE CONTINUAT DIN TASK 2 - sapt 3 : rejoin 
"""TASK 3 - :  Rejoin stemmed tokens and use a TfIdf Vectorizer on the descriptions"""

def rejoin_stemmed_words(stemmed_words):
    return [' '.join(words) for words in stemmed_words]

# stemmed_words = stem_tokens()   # same as stemming called above 
# joined_words = rejoin_stemmed_words(stemmed_words)
# print(joined_words)

# tfidf = TfidfVectorizer(tokenizer=stemming, stop_words='english')    
# y = tfidf.fit_transform()
# print(tfidf.get_features_names_out())
# print(y.shape())


"""  TASK 4 - Applying a Tfidf Vectorizer on the raw text from description field >"""
data_raw = []
for doc in collection.find({}):
    data_raw.append(doc['description'])

tfidf_vectorizer = TfidfVectorizer()    
tfidf_raw = tfidf_vectorizer.fit_transform(data_raw)
# print("Raw text:")
# print(tfidf_raw.toarray())
# Initialize the TfidfVectorizer with the desired settings
vectorizer = TfidfVectorizer()
# Fit the vectorizer to the list of stemmed descriptions
tfidf_matrix = vectorizer.fit_transform(snowballStem)   # "stemming" for Porter Stemmer

print(tfidf_matrix.shape)       # the shape for the tfidf document (for the prontos.json)       -- Returns : (2637,5392)
# tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vocabulary)
# print(tfidf_df)

feature_names = vectorizer.get_feature_names_out()
tfidf_score = tfidf_matrix.toarray()[0]

# Print the feature names and their corresponding TF-IDF scores if it is a nonzero score in the description 
# for i in range(len(feature_names)):
#     if tfidf_score[i] > 0:
#         print(f"{feature_names[i]}: {tfidf_score[i]}")

def tfidf_scores(stemming):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(lancaster)   # this was stemming instead of lancaster

    feature_names = vectorizer.get_feature_names_out()
    tfidf_score = tfidf_matrix.toarray()[0]

    tfidf_dict = {}
    for i in range(len(feature_names)):
        if tfidf_score[i] > 0 :
            tfidf_dict[feature_names[i]] = tfidf_score[i]

    return tfidf_dict

""" print(tfidf_scores(stemming)) """

# Example :Get the vocabulary (i.e. the unique terms in the corpus) and the idf values
# vocabulary = vectorizer.get_feature_names_out()
# idf_values = vectorizer.idf_
# print("Vocabulary : ", vocabulary)
# print("IDF values : " , idf_values)

# TASK WEEK 4
""""Title and description of a PR should be concatenated and then tokenized, stemmed, etc."""
stop_words = set(stopwords.words('english'))
porter = PorterStemmer()
snowball = SnowballStemmer(language="english")
lancaster = LancasterStemmer()
def tokenize_text(text):
    # Tokenize the text and remove stopwords
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    return filtered_tokens

def stem_tokens(tokens):
    # Stem the tokens using PorterStemmer
    stemmed_tokens = [lancaster.stem(token) for token in tokens]
    return stemmed_tokens


# for this to get a better accuracy we should go above to the stem_tokens() where we used PorterStemmer and use SnowballStemmer (maybe will improve the accuracy)
def concat_full(database_test, collection_test):
    db = client[database_test]
    collection = db[collection_test]
    data = []
    for pr in collection.find():
        pr_text = pr["title"] + " " + pr["description"] # concatenate the title and description fields and use tokenize on it and then stemming 
        tokens = tokenize_text(pr_text)
        stemmed_tokens = stem_tokens(tokens)
        processed_text = ' '.join(stemmed_tokens)
        data.append({"_id": str(pr["_id"]), "processed_text_v2" : processed_text})
        collection.update_one({"_id": pr["_id"]},{"$set": {"processed_text_v2": processed_text}})
    
    dict_vectorizer = DictVectorizer()
    sparse_matrix = dict_vectorizer.fit_transform(data)
    # print(sparse_matrix)
    return sparse_matrix.toarray()        # Sparse matrix on this concat_full() function

text_concatenate = concat_full('database_test', 'collection_test')
# print(text_concatenate)

""" have to do sparsematrix - on this function ( vectorize) like features_extract()"""

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
    # print(feature_list)        # printing the list of all the features that we wanna see , for example :  'build' & 'feature' from the json file
    dict_vectorizer = DictVectorizer()
    sparse_matrix = dict_vectorizer.fit_transform(feature_list)
    # print(sparse_matrix)        # printing the sparse_matrix for each stuff inside the fields
    return sparse_matrix.toarray()
# , dict_vectorizer.get_feature_names_out()

features = features_extract('database_test', 'collection_test')
# print(features)

"""then i have to use train and testset for these 2 functions to get what we need  """


# Concatenate the TFIDF results with the other features extracted

# Turn our categorical groups in charge into numbered fields (LabelEncoder)     -- field : groupInCharge --> LabelEncoder()

from sklearn.preprocessing import LabelEncoder
def encode_groupInCharge(database_test,collection_test, field_name):
    db = client[database_test]
    collection = db[collection_test]
    documents = collection.find({})
    
    # Encode the 'groupInCharge' field using LabelEncoder()
    field = [doc[field_name] for doc in documents]
    encoder = LabelEncoder()    
    encoded_labels = encoder.fit_transform(field)
    df = pd.DataFrame({'groupInCharge' : field, 'encoded_labels': encoded_labels})
    # Return the encoded labels as a dataframe to see in detail the encoded_label value for each content in groupInCharge 
    return df

encoded_labels = encode_groupInCharge('database_test', 'collection_test', 'groupInCharge')
# print(encoded_labels)

# Encoding with LabelEncoder for 'state' field ( to have only 2 values ( Closed , Correction Not Needed ) not more than 2 like 'groupInCharge')

def encode_state(database_test, collection_test, field_name):
    db = client[database_test]
    collection = db[collection_test]

    documents = collection.find({})

    #Encode 'state' field using LabelEncoder()
    field = [doc[field_name] for doc in documents]
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(field)
    df = pd.DataFrame({'state' : field, 'encoded_labels' : encoded_labels})
    # Return the encoded labels as a dataframe to see in detail the encoded_label value for each content in 'state'
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
    print("The accuracy is : ", score)
    print("Accuracy: {:.2f}%".format(score * 100))
    return clf

splitting_data = splitData('database_test', 'collection_test', 0.2)



"""WEEK 6"""
# Concatenare sparse matrix la cele 2 functii de mai jos cu HSTACK : https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.hstack.html

# Train si test data se poate folosi pe cele 2 matrici concatenate cu ajutorul hstack-ului https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.hstack.html
# LabelEncoder() pentru valorile ptr BOAM sau NOT BOAM ( MANO )    - o functie separata cu ceva IF is not BOAM == .. s.a.m.d
# Fill -- numpy.nan_to_num pentru a inlocui valorile lipsa cu ceva default i guess

# Encoding the categoric data from 'groupInCharge' field into labels
def label_encodeGIC(database_test,collection_test):
    gic_list = []

    cursor = collection.find({})
    for doc in cursor:
        gic = doc['groupInCharge'].split('_', 1)[0] # getting the first word before the '_' in the groupInCharge field
        # print(gic)
        if(gic != 'MANO' or gic !='BOAM'):
            gic_list.append('not_BOAM')
        else: 
            if gic == 'MANO':
                gic_list.append('BOAM')
            else:
                gic_list.append(gic)
    label_encoder = LabelEncoder()
    gic_encoded = label_encoder.fit_transform(gic_list)
    # print(gic_encoded)
    return gic_encoded

encodingGIC = label_encodeGIC('database_test','collection_test')
# print(encodingGIC)

# Concatenare sparse matrixes la cele 2 functii (concat_full(), feature_extract()) cu HSTACK : https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.hstack.html

def concatenate(database_test, collection_test):
    feature_extract = features_extract(database_test, collection_test)      # (2637,2460)
    concat_text = concat_full(database_test, collection_test)   # this is (2637, 5273)

    print("Features_extract : ", feature_extract.shape)
    print("Concat_full : ", concat_text.shape)

    # Reshape concat_text to have shape (2637, 5273)
    concat_text_reshaped = np.reshape(concat_text, (concat_text.shape[0], concat_text.shape[1]))    # reshaping the concat_full() to have the same value with features_extract() for hstacking
    #Convert concat_text_reshaped to a sparse matrix
    concat_text_sparse = scipy.sparse.csr_matrix(concat_text_reshaped)

    # Concatenate feature_extract and concat_text_sparse horizontally
    full_concatenate = hstack([feature_extract, concat_text_sparse])
    scipy.sparse.save_npz('sparse_matrix.npz', full_concatenate)
    print(full_concatenate.toarray()) 
    print(full_concatenate.shape)   # the shape for this concatenate function : (2637,7733)
    return full_concatenate

concatenated_sparse_matrices = concatenate('database_test','collection_test')
# print("Concatenated Sparse Matrices: ", concatenated_sparse_matrices)



# DecisionTreeClassifier for this concatenate() function which have 2 sparse matrices concatenated 

from sklearn.impute import SimpleImputer
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
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    #Test the model on the testing set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,  y_pred)
    print('Accuracy: {:.2f}%'.format(accuracy * 100))
    return clf


# clf = train_decision_tree('database_test', 'collection_test')

np.set_printoptions(threshold=np.inf)


# modify this to exclude not_state values but still get a null value for the array
def label_encodeState(database_test,collection_test):
    
    state_list = []
    cursor = collection.find({})
    for doc in cursor:
        state = doc['state'] # setting a variable with 'state' field items
        if(state != 'Closed' and state !='Correction Not Needed'):
            state_list.append('Closed')
        else: 
            if state == 'Closed':
                state_list.append('Closed')
            else:
                state_list.append('Correction Not Needed')
        # if(state == 'Closed' and state !='Correction Not Needed'):
        #     state_list.append('Closed')
        # elif(state == 'Correction Not Needed'):
        #     state_list.append('Correction Not Needed')


    count_nonzero = np.count_nonzero(state_list)
    print("Number of non-zero elements in state_list:", count_nonzero)

    # print(state_list)
    label_encoder = LabelEncoder()
    encoded_states = label_encoder.fit_transform(state_list)

    # print(encoded_states)
    return encoded_states

encodingState = label_encodeState('database_test','collection_test')

# modify algorithm to get an accuracy more than 70%
# Using a Naive Bayes Classifier 

# from sklearn.naive_bayes import GaussianNB
# def gaussian_NB(database_test, collection_test):
#     feature_vect = scipy.sparse.load_npz('sparse_matrix.npz').toarray()
#     feature_vect =  np.nan_to_num(feature_vect, copy=False, posinf=0.0, neginf=0.0)  # values of 0.0
#     target = label_encodeState(database_test, collection_test)

#     X_train, X_test, y_train, y_test = train_test_split(feature_vect, target)

    
#     if X_train.shape[0] != y_train.shape[0]:
#         raise ValueError("Number of samples in X_train and y_train do not match.")

#     naiveBayes = GaussianNB()
#     predict_target = naiveBayes.fit(X_train, y_train).predict(X_test)
#     # print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != predict_target).sum()))
#     accuracy = accuracy_score(y_test,  predict_target)
#     print('Accuracy: {:.2f}%'.format(accuracy * 100))       # 100% accuracy because is something wrong with label_encodeGIC  with MANO AND BOAM , and i can change the target with state field ( closed and CorrectionNotNeeded)

if __name__ == "__main__":
    # gaussian_NB('database_test', 'collection_test')
    train_decision_tree('database_test', 'collection_test')

client.close()