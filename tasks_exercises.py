from pymongo import MongoClient 
import pymongo
import pprint 
import json

with open("prontos.json", "r") as data:
    prontos = json.load(data)

client = MongoClient("localhost", 27017)

db = client["database_test"]    #  creating a database inside the mongodb client
collection = db["collection_test"]  #then create a new collection in that database

def possible_groupsInCharge(client:MongoClient):
    count = client["database"]["collection"].count_documents(filter={}) 
    grp_docum = {}
    grp_list = client["database"]["collection"].distinct("groupInCharge")   # for every distinct values in the "groupInCharge" from the collection

#Iterate through each document in that collection "groupInCharge" attribute
    for el in grp_list:
        grp_docum[el] = client["database"]["collection"].count_documents(filter={"groupInCharge": el}) / count * 100    # getting the percentage of "groupInCharge"
    sorted_group = sorted(grp_docum.items(),key=lambda x : x[1], reverse=True) # sorting in a descending order 
    return sorted_group


#A function to return all the features that were tested 
def featureList_tested(client:MongoClient):
    feature = client["database"]["collection"].distinct("feature")
    # pprint.pprint(feature)
    return feature

#List all possible pronto resolutions(STATE) and percentage of occurance for each
# Resolution of the pronto means the status of each individual ( STATE in the json file)
def pronto_resolution(client:MongoClient):
    count_prontos = client["database"]["collection"].count_documents(filter={})
    grp_prontos = {}
    pronto_list = client["database"]["collection"].distinct("state")
    for elem in pronto_list:
        grp_prontos[elem] = client["database"]["collection"].count_documents({"state" : elem}) / count_prontos * 100  # to show the result in procentage for each stat of the pronto
    sorted_prontos = sorted(grp_prontos.items(), key=lambda y : y[1], reverse=False) # sorting in ascending order
    return sorted_prontos


#Sort prontos by number of attachments : "attachedPRs"
def sorted_numberAttachments(client:MongoClient):
    list_attach = client["database"]["collection"].find({}, {"_id": 1, "attachedPRs": 1}).sort([("attachedPRs", pymongo.ASCENDING)])
    new_listAttach = [var for var in list_attach]
    new_listAttach.sort(key=lambda y: len(y.get("attachedPRs", [])))
    return new_listAttach


#Sort prontos by number of information requests
def information_request(client:MongoClient):
    inform_list = client["database"]["collection"].find({},{"_id": 1, "informationrequestID": 1}).sort([("informationrequestID", pymongo.ASCENDING)])
    new_info = [var for var in inform_list]
    new_info.sort(key=lambda x: len(str(x["informationrequestID"])))
    return new_info
#Correction not needed 
def corrNotNeeded(client:MongoClient):
    var_state = client["database"]["collection"].count_documents({"state": "Correction Not Needed"})
    var_count = client["database"]["collection"].count_documents(filter={})
    correctionNN = (var_state / var_count) * 100    # Showing the result for the "CorrectionNotNeeded" attribute in procentage
    pprint.pprint(f"Percentage of Correction Not Needed is : {correctionNN:.5f} %")


# pprint.pprint(possible_groupsInCharge(client))
# pprint.pprint(featureList_tested(client))
# pprint.pprint(corrNotNeeded(client))
# pprint.pprint(information_request(client))
# pprint.pprint(sorted_numberAttachments(client))
pprint.pprint(pronto_resolution(client))

client.close()