# """ Natural Language Toolkit  - nltk in python"""

# """ Simple things to do using NLTK"""
# import nltk
# from nltk.tokenize import word_tokenize
# import json
# from pymongo import MongoClient

# # nltk.download('punkt')
# # sentence =  """ At eight o'clock on Thursday morning Arthur didn't feel so good."""
# # tokens = nltk.word_tokenize(sentence)
# # print(tokens)

# client = MongoClient("localhost", 27017)

# with open("prontos.json", "r") as data_file:    
#     data = json.load(data_file)


# description = nltk.word_tokenize(data)
# print("Data is : \n ", description)

#Checking if the 'description' field is an instance of either a string or bytes object before tokenizing it

# Function to tokenize a specific field in a database 
# def desc_tokenize(client):
#     db = client['database']
#     collection = db['collection']

#     tokens_list = []
#     for doc in collection.find():
#         if isinstance(doc['description'], (str, bytes)):
#             print(doc['description'])
#             tokens = word_tokenize(doc['description'])
#             print(tokens)
#             tokens_list.append(tokens)
#     return tokens_list



# client.close()




# # tagged = nltk.pos_tag(tokens)
# # print(tagged[0:7])

# # entities = nltk.chunk.ne_chunk(tagged)
# # print(entities)