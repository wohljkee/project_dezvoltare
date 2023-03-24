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


"""Method 1 v1"""
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

# Initialize the TfidfVectorizer with the desired settings
vectorizer = TfidfVectorizer()
# Fit the vectorizer to the list of stemmed descriptions
tfidf_matrix = vectorizer.fit_transform(stemming)

print(tfidf_matrix.shape)       # the shape for the tfidf document (for the prontos.json)
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

print(tfidf_scores(stemming))



#Get the vocabulary (i.e. the unique terms in the corpus) and the idf values
# vocabulary = vectorizer.get_feature_names_out()
# idf_values = vectorizer.idf_
# print("Vocabulary : ", vocabulary)
# Print the vocabulary and idf values
# print("Vocabulary : " , vocabulary)
# print("IDF values : " , idf_values)

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



# TASK WEEK 4
""""Title and description of a PR should be concatenated and then tokenized, stemmed, etc."""

# def process_pr_text(database_test, collection_test):
#     db = client[database_test]
#     collection = db[collection_test]

#     for pr in collection.find():
        
#         pr_text = pr["title"] + " " + pr["description"] # concatenate the title and description fields and use tokenize on it and then stemming 
#         tokens = word_tokenize(pr_text)
#         # stemmed_tokens = [PorterStemmer().stem(token) for token in tokens]
#         processed_text = ' '.join(tokens)
#         collection.update_one({"_id": pr["_id"]},{"$set": {"processed_text": processed_text}})
#     return processed_text

# process_pr_text("database_test", "collection_test")

#Turn the other data into useful features for our model (DictVectorizer, OneHotEncoder)
# Getting from the pronto : 'build' , 'feature' fields for now
# Using DictVectorizer

# def features_extract(database_test, collection_test):
#     db = client[database_test]
#     collection = db[collection_test]

#     feature_list = []
#     for doc in collection.find():
#         features = {} 
#         features['feature'] = doc.get('feature', '')
#         features['build'] = doc.get('build', '')
#         feature_list.append(features)
    
#     dict_vectorizer = DictVectorizer
#     feature_matrix = dict_vectorizer.fit_transform(feature_list)

#     return feature_matrix, dict_vectorizer.get_feature_names_out()


client.close()

