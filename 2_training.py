#Mounting the Google Drive
data_root='data'
#Please upload the files in your drive and change the path to it accordingly.
# nltk.download("punkt")
# nltk.download ("wordnet")

#2 Importing Relevant Libraries
import json
import string
import random
import sys
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint

from classCat.a_04_model import MyClassModel,BackupModelCheckpoint
from classCat.d_12_library import pred_class, preprocess_data

words=[] #For Bow model/ vocabulary for patterns
classes= [] #For Bow model/ vocabulary for tags

#3 Loading the Dataset: intents.json
data_file = open(data_root+'/combined.json').read()
data = json.loads(data_file)

# print(data)
# sys.exit(0)
#4 Creating data_X and data_Y
data_X= [] #For storing each pattern
data_y= [] #For storing tag corresponding to each pattern in data_X
words, classes, data_X, data_y = preprocess_data(data, data_X, data_y)

# print(classes)
lemmatizer= WordNetLemmatizer()

#5 Text to Numbers
training= []
out_empty= [0]*len(classes)
# creating the bag of words model
for idx, doc in enumerate(data_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    #mark the index of class that the current pattern is associated
    # to
    output_row= list(out_empty)
    output_row[classes.index(data_y[idx])] = 1
    #add the one hot encoded Bowl and associated classes to training
    training.append([bow, output_row])

#shuffle the data and convert it to an array
random.shuffle(training)
training= np.array(training, dtype=object)
#split the features and target labels 
train_X= np.array(list(training[:, 0]))
train_Y= np.array(list(training[:, 1]))


#6 The Neural Network Model
# Usage example:
# Instantiate the model
num_classes = len(train_Y[0])
model = MyClassModel(num_classes)

# Add layers
model.add_layer(1024, activation='relu', input_shape=(len(train_X[0]),))
model.add_layer(512, activation='relu', dropout_rate=0.5)
model.add_layer(num_classes, activation='softmax')

# Build the model
model.build_model(input_shape=(len(train_X[0]),))

# Compile the model
adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Print model summary
print(model.summary())
# print (model.summary())

model.load_weights('path_to_saved_model/variables/variables')

# Create the BackupCallback instance
# Define the checkpoint filepath for backup
checkpoint_filepath = 'backup_weights'
checkpoint_callback = BackupModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, backup_frequency=20)

model.fit(x=train_X, y=train_Y, epochs=80,batch_size=64, verbose=0,callbacks=[checkpoint_callback], validation_data=(train_X, train_Y))
model.save('path_to_saved_model')


#7 Preprocessing the Input
def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that. Can you please provide more information?"
    
    tag = intents_list[0]
    list_of_intents = intents_json.get("intents", [])
    
    for intent in list_of_intents:
        if intent.get("tag") == tag:
            return random.choice(intent.get("responses", []))
    
    return "Sorry, I'm not sure how to respond to that."

def chat():
    print("Type 'FUCK' if you don't want to chat with our ChatBot.")
    while True:
        message = input("").lower()
        if message == "fuck" or message == "end":
            break
        intents = pred_class(message, words, classes, model)
        result = get_response(intents, data)
        print(result)

if __name__ == "__main__":
    chat()






    