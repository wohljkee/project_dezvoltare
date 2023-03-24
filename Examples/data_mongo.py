import pymongo
from pymongo import MongoClient

connection_string = "mongodb://localhost:27017/appdb"   # connecting my linux ip ( or localhost ) with the port of the mongodb 27017, and the name of the database is 'database"

client = pymongo.MongoClient(connection_string)

# client.get_database()
print(client.list_database_names())     # printing all the databases already existing
# print(client)

db = client.get_database("appdb")  #getting the database already created "appdb"

print(db.list_collection_names())

collection = db.get_collection("bikes") # creating a new collection 

# document = {"name":"Honda",
#             "color":black,
#             "year_release":2005,
# }
documents = []
documents.append(  {"name":"Honda",
                    "color":"black",
                    "year_release":2005,
                    "noise": "25dB"
                })

documents.append({"name":"Suzuki",
                     "color":"white",
                    "year_release":2008,
                      "noise": "30dB"
                })

documents.append({"name":"Kawasaki",
                    "color":"green",
                    "year_release":2020,
                    "noise": "27dB"
                }
                )

response = collection.insert_many(documents) # adding a new document in the database

last_inserted_ids = response.inserted_ids
print(f"Last inserted id is : {last_inserted_ids}")
# print(document)
# print(collection.find())




