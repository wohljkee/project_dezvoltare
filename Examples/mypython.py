# -*- coding: utf-8 -*-
import json
import pandas as pd # importing pandas library 

# Load the JSON file into a Python dictionary
with open("ex_prontos.json", 'r') as data_file:
    # data_content = data_file.read()
    data = json.load(data_file)

df = pd.json_normalize(data)

# Select the key that we want to see stuff for it , for example groupInCharge in this case
print(list(data['groupInCharge']))
# group_column = df["groupInCharge"]
# print(group_column)
# Calculate the percentage of the groupinCharge in the data : 

# TASK  : list all possible groups in charge and percentage of occurance for eachgr
# Need a function to show all the groups , and another function to give us the procentage of the group



# """# Creating an empty dictionary to store the group   names and their occurances
# group_counts = {}

# #Iterate through the dictionary and count the occurances of each group name
# for item in data:
#     if item["groupInCharge"] in item:
#         group_name = item["groupInCharge"]
#         if group_name:
#             if group_name in group_counts:
#                 group_counts[group_name] += 1
#             else:
#                 group_counts[group_name] = 1


# #calculate the percentage of occurances of each group name:
# total_items = len(data)
# for group_name, count in group_counts.items():
#     percentage = count / total_items * 100
#     group_counts[group_name] = percentage

# # Print out the list of group names and their occurances percentages
# for group_name, percentage in group_counts.items():
#     print(f"{group_name}: {percentage:.2f}%")"""



# print(data_content) # printing the ex_prontos.json file in the console

# parsed_json = json.loads(data_content)
# df = pd.read_json("ex_prontos.json")
# print(df.head()

def calc_nrOfgroups(my_list):
    #Creating an empty dictionary
    dict = {}
    for item in my_list:
        return

