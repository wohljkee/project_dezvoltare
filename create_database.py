import pymongo
from pymongo import MongoClient

connection_string = "mongodb://localhost:27017/appdb"   # connecting my linux ip ( or localhost ) with the port of the mongodb 27017, and the name of the database is 'database"

client = pymongo.MongoClient(connection_string)

# client.get_database()
print(client.list_database_names())     # printing all the databases already existing
# print(client)

db = client.get_database("database_project")  #creating a new database named "database_project"

print(db.list_collection_names())

collection = db.get_collection("bikes") # creating a new collection 

"""https://www.w3schools.com/python/python_mongodb_query.asp
https://www.w3schools.com/python/python_mongodb_create_collection.asp"""

