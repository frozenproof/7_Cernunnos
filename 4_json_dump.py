import json
import os

data_root='data'
dataset_root='datasets'

# Initialize an empty list to store combined intent arrays
combined_intents = []

# Iterate through each file in the folder
for filename in os.listdir(dataset_root):
    if filename.endswith('.json'):
        # Open the JSON file
        with open(os.path.join(dataset_root, filename), 'r') as file:
            data = json.load(file)
        
        # Append the intents to the combined list
        combined_intents.extend(data['intents'])

# Write the combined intents to a new JSON file
combined_data = {'intents': combined_intents}
with open(data_root+'/combined.json', 'w') as combined_file:
    json.dump(combined_data, combined_file, indent=4)