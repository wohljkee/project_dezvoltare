""" Natural Language Toolkit  - nltk in python"""

""" Simple things to do using NLTK"""
import pprint
import nltk
import json
import pandas as pd
from nltk.tokenize import word_tokenize
from pymongo import MongoClient
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

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
releas rf softwar current lte primari link expect gnb sw activ gnb make lte actual gnb sw activ gnb make radio modul make reset lte traffic tester analysi log actionportchang eventportchang trigg
er swbot inform gic pronto group charg given file name contain indic file timestamp fault found attach need document specif made fault report rf share use use enabl ssh even tr btslog disabl de
bug info warn error vip disabl none udp sic standard output local remot valid aasyslogoutputaddress aasyslogudpport bbc debug flag fault occurr mani time test scenario mani time fault mani site
live oper run case custom test scenario histori chang sinc test success last test scenario pass ye last sw version test scenario differ sinc last time test scenario ye chang test scenario sinc
last run one sector configur test case rp ute need specif made fault report custom made ticket end default templat'
"""

"""Method 1 v2"""
# def stem_tokens():
#     ps = PorterStemmer()
#     stems_list = [] # this is a empty list to append all the stemmed data
#     db = client["database_test"]
#     collection = db["collection_test"]
#     for doc in collection.find({}):
#         description = doc['description']
#         stem = [ps.stem(tok) for tok in description.split() if tok.isalpha()] 
#         collection.update_one({"_id": doc["_id"]}, {'$set': {'tokenized_description': stem}})
#         stems_list.append(stem)
#     return stems_list

# stemming = stem_tokens()

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
tfidf_matrix = vectorizer.fit_transform(stemming)

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
    tfidf_matrix = vectorizer.fit_transform(stemming)

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

def tokenize_text(text):
    # Tokenize the text and remove stopwords
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    return filtered_tokens

def stem_tokens(tokens):
    # Stem the tokens using PorterStemmer
    stemmed_tokens = [porter.stem(token) for token in tokens]
    return stemmed_tokens


def concat_full(database_test, collection_test):
    db = client[database_test]
    collection = db[collection_test]
    for pr in collection.find():
        pr_text = pr["title"] + " " + pr["description"] # concatenate the title and description fields and use tokenize on it and then stemming 
        tokens = tokenize_text(pr_text)
        stemmed_tokens = stem_tokens(tokens)
        processed_text = ' '.join(stemmed_tokens)
        collection.update_one({"_id": pr["_id"]},{"$set": {"processed_text": processed_text}})
    return processed_text

text_concatenate = concat_full('database_test', 'collection_test')

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
    print(feature_list)        # printing the list of all the features that we wanna see , for example :  'build' & 'feature' from the json file
    dict_vectorizer = DictVectorizer()
    sparse_matrix = dict_vectorizer.fit_transform(feature_list)
    print(sparse_matrix)        # printing the sparse_matrix for each stuff inside the fields
    return sparse_matrix.toarray()
# , dict_vectorizer.get_feature_names_out()

features = features_extract('database_test', 'collection_test')

print(features)

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
print(encoded_labels)


client.close()

