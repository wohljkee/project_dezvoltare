import json
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder


def extract_and_transform_fields(filename, fields):
    """ Extracts the specified fields from a JSON file and transforms them using DictVectorizer and OneHotEncoder.
         Parameters : - filename (str) - the name of the JSON file
                      - fields(list of str) -  the name of the fields to extract and transform
    
        Returns : X(numpy array): The transformed feature matrix
    """

    # Load the JSON data:
    with open(filename,'r') as f:
        data = json.load(f)

    # Extract the specific fields from the data ( json file )
    # extracted_data = []
    # for item in data:
    #     extracted_item = {field: item[field] for field in fields}
    #     extracted_data.append(extracted_item)
    extracted_data = []
    for item in data:
        extracted_item = {}
        for field in fields:
            # Check whether the value can be converted to an integer
            value = item[field]
            if isinstance(value, str) and value.isdigit():
                value = int(value)
            extracted_item[f"{field}={value}"] = 1
        extracted_data.append(extracted_item)


    # Convert the extracted data to a Pandas dataframe 
    df = pd.DataFrame(extracted_data)

    # Instantiate the DictVectorizer
    vect = DictVectorizer(sparse=False) 
    
    # Transform the data using DictVectorizer
    X = vect.fit_transform(df.to_dict('records'))

    # Instantiate the OneHotEncoder
    enc = OneHotEncoder()

    # Transform the data using the OneHotEncoder

    X = enc.fit_transform(X)

    # Print the feature names :
    print(vect.get_feature_names())

    # Return the transformed feature matrix for those 2 types of vectorizing
    return X

# X = extract_and_transform_fields('ex_prontos.json', ['build', 'feature'])
# print(X)



# Training and testing a dataset
import pandas as pd
from sklearn.datasets import load_iris

iris_data = load_iris()
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
print(df)

training_data = df.sample(frac=0.8, random_state=25)    # 0.8 = 80%
testing_data = df.drop(training_data.index)

print(f"No. of training examples : {training_data.shape[0]}")
print(f"No. of testing examples : {testing_data.shape[0]}")
