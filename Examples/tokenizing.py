import nltk
from nltk.tokenize import word_tokenize
import json
from pymongo import MongoClient
client = MongoClient("localhost", 27017)

with open("prontos.json", "r") as data_file:    
    data = json.load(data_file)

db = client["database"]    #  creating a database inside the mongodb client
collection = db["collection"]  #then create a new collection in that database

def tokenize_words(client:MongoClient):
    db = client["database"]    #  creating a database inside the mongodb client
    collection = db["collection"]
    elem = collection.find({})
    for doc in elem:
        if isinstance(doc["description"], (str, bytes)):
            descr_token = doc["description"]
            tokens = word_tokenize(descr_token)
            collection.update_one({"_id": doc["_id"]}, {'$set': {"description": tokens}})
    return tokens


tokenizerr = tokenize_words(client)