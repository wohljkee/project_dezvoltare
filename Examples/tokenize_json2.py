""" Natural Language Toolkit  - nltk in python"""

""" Simple things to do using NLTK"""
import pprint
import nltk
from nltk.tokenize import word_tokenize
import json
from pymongo import MongoClient, UpdateOne 
from nltk.stem import PorterStemmer, SnowballStemmer

client = MongoClient("localhost", 27017)

with open("prontos.json", "r") as data_file:    
    data = json.load(data_file)

db = client["database"]    #  creating a database inside the mongodb client
collection = db["collection"]  #then create a new collection in that database

""" TASK 1"""

def desc_tokenize(client:MongoClient):
    db = client['database']
    collection = db['collection']
    elem = collection.find({})
    tokens_list = []    # creating an empty array for the tokenized data 
    for doc in elem:  # for every document inside the collection 
        # print(doc["description"])
        if isinstance(doc["description"], (str, bytes)):        # verify if the 'description' field from the collection is of type "string or bytes"
            tokens = word_tokenize(doc["description"])  # used the word_tokenize() method on the ['description'] field to tokenize every words in that field
           
            # joined_tokens = ' '.join(tokens)
            tokens_list.append(tokens)  # here we added everything that was tokenized in the empty array we created
            collection.update_one({"_id": doc["_id"]}, {'$set': {"description": tokens}})   # here we updated the collection for each string in the 'description' field
            
    return tokens_list  

tokens = desc_tokenize(client)

"""TASK 2  :: Apply different stemming methods on the tokens """

ps = PorterStemmer()

"""Method 1"""
def stem_tokens(token):
    stems_list = []
    db = client["database"]
    collection = db["collection"]
    for token in collection.find({}):
        description = token['description']
        stem = [ps.stem(word) for word in description if word.isalpha()] 
        stems_list.append(stem)
        collection.update_one({"_id": token["_id"]}, {'$set': {"description": stem}})
    return stems_list

stem_tokens(tokens)


client.close()