from pymongo import MongoClient 
import pymongo
import pprint   # PRETTY PRINT
import json
""""values": [
    {
      "problemReportId": "PR493986",
      "faultAnalysisId": [
        "FA562744"
      ],
      "attachedPRs": [],
      "author": "Steta, Mihail (Nokia - RO/Timisoara)",
      "build": "SBTS19B_ENB_0000_000827_000000",
      "description": "*** DEFAULT TEMPLATE for 2G-3G-4G-5G-SRAN-FDD-TDD-DCM-Micro-Controller common template v1.1.0 (09.05.2018) – PLEASE FILL IT IN BEFORE CREATING A PR AND DO NOT CHANGE / REMOVE ANY SECTION OF THIS TEMPLATE ***\r\n\r\n[1. Detail Test Steps:]\r\n1.SWR started\r\n2. Check SBTS state after software replacement\r\n\r\n[2. Expected Result:]\r\n\r\n1. SBTS on air\r\n2. All cells on air\r\n3. No unnexpected alarms present\r\n\r\n[3. Actual Result:]\r\n1. Ok\r\n2. NOK . 2 WCDMA cells and one LTE cell are down\r\n\r\n[4. Tester analysis:]\r\n\r\nAnalysis of Logs: 2 WCDMA cells and on LTE cell are down after Softare replacement. All cells are mapped on the same RFM, FRGB type. \r\n\r\nNo alarms raised, nothing relevant found in traces despite the issue is permanent.\r\n\r\nAction performed to recover:\r\n- after RFM reset - same isue;\r\n- after RFM block/unblock - same issue;\r\nOnly SBTS reset solve the issue, all cells are on air.\r\n\r\nSWBOT information:\r\nMANO_WRO_HWMGMT 40%\r\nMANO_TIM_FMRECOV 40%\r\n\r\n\r\n\r\n\r\n[5. Log(s) file name containing a fault: (clear indication (exact file name) and timestamp where fault can be found in attached logs)]\r\n\\\\eseefsn50.emea.nsn-net.net\\rotta4internal\\HetRAN\\msteta\\Cells down after SWR from 19A to 19B\r\n\r\n[6. Test-Line Reference/used HW/configuration/tools/SW version]\r\nSBTS10152\r\n1LW2_1.1.1_FRGY_1x6G_IA_1_444;1LW2_1.1.2_FRGY_1x6G_IA_1_444;1LW2_1.1.3_FRGY_1x6G_IA_1_444;B1FLW_1+B\r\n\r\nBase build: SBTS19A_ENB_0000_000229_445911 \r\nTarget build: SBTS19B_ENB_0000_000827_000000\r\n\r\n[7. Used Flags: (list here used R&D flags)]\r\nun swconfig\r\n\r\n[8. Fault Occurrence Rate:]\r\n   How many times Test Scenario was run?\r\n1\r\n   How many times fault was reproduced?\r\n1\r\n   How many sites in the same live operation was run in case of customer fault? \r\nN/A \r\n\r\n[9. Test Scenario History of Execution: (what was changed since it was tested successfully for the last time)]\r\n   Was Test Scenario passing before?\r\nyes\r\n   What was the last SW version Test Scenario was passing?\r\nSBTS19B_ENB_0000_000734_000000\r\n   Were there any differences between test-lines since last time Test Scenario was passing?\r\nno\r\n   Were there any changes in Test Scenario since last run it passed?\r\nno\r\n[10. Test Case Reference: (QC, RP or UTE link)]\r\nnot needed in documentation, specification made fault reports and customer made tickets\r\n\r\n*** END OF DEFAULT TEMPLATE ***",
      "feature": "System_Operability",
      "groupInCharge": "MANO_MNL_RADIOCTRL",
      "state": "Correction Not Needed",
      "title": "[SBTS19B][CIT][FSM] 2 WCDMA cells and one LTE cell are down after SWR from SBTS19A to SBTS19B",
      "authorGroup": "NITSIVBTS8",
      "informationrequestID": [
        "IR158069"
      ],
      "statusLog": null,
      "release": [
        "SBTS19B"
      ],
      "explanationforCorrectionNotNeeded": [
        "Pronto was not reproducible with same build in IR request by FRI. Author agreed to set PR as CNN-FNR"
      ],
      "reasonWhyCorrectionisNotNeeded": [
        "Fault was not reproducible"
      ],
      "faultAnalysisFeature": [],
      "faultAnalysisGroupInCharge": [
        "MANO_MNL_RADIOCTRL"
      ],
      "stateChangedtoClosed": null,
      "faultAnalysisTitle": [
        "[SBTS19B][CIT][FSM] 2 WCDMA and one LTE cells remaind down after SWR from SBTS19A to SBTS19B failed"
      ]
    }
"""
with open("prontos.json", "r") as data:
    prontos = json.load(data)

client = MongoClient("localhost", 27017)

db = client["database_test"]    #  creating a database inside the mongodb client
collection = db["collection_test"]  #then create a new collection in that database
# collection.insert_many(prontos)


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
    # inform_list = client["test-database"]["collection"].find({})
    # new_info = []
    # for var in sorted(inform_list, key=lambda x: len(x["informationrequestID"])):
    #     new_info.append(var)
    
    # return new_info

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