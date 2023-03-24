from pymongo import MongoClient # importing MongoDB client to write to a database
import json # library used to read a json file

client = MongoClient("localhost", 27017)

db = client["database_test"]    #  creating a database inside the mongodb client
collection = db["collection_test"]  #then create a new collection in that database

"""Opening a json file with the prontos data inside"""
with open("prontos.json", "r") as data_file:    
    data = json.load(data_file)

listOfValues = data["values"] #creating a variable with the data "values" inside of it

collection.insert_many(listOfValues) # adding that list of variables in the database collection

print(collection.find_one())    #Finding the first document in the customers collection for example


client.close() # closing client afterwards  
