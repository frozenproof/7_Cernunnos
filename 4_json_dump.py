import json
import os

data_root = 'data'
dataset_root = 'datasets'

# Initialize an empty dictionary to store combined intents
combined_intents = {}

# Iterate through each file in the folder
for filename in os.listdir(dataset_root):
    if filename.endswith('.json'):
        # Open the JSON file
        with open(os.path.join(dataset_root, filename), 'r') as file:
            data = json.load(file)
        
        # Iterate through intents in the current file
        for intent in data['intents']:
            tag = intent['tag']
            # Check if tag already exists in combined data
            if tag in combined_intents:
                # If tag exists, merge contents
                existing_intent = combined_intents[tag]
                existing_intent['patterns'].extend(intent['patterns'])
                existing_intent['responses'].extend(intent['responses'])
            else:
                # If tag doesn't exist, add it to combined data
                combined_intents[tag] = intent

# Convert dictionary values to a list
combined_data = {'intents': list(combined_intents.values())}

# Write the combined intents to a new JSON file
with open(os.path.join(data_root, 'combined.json'), 'w') as combined_file:
    json.dump(combined_data, combined_file, indent=4)
