import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from keras.callbacks import Callback
import tensorflow as tf

lemmatizer= WordNetLemmatizer()

import nltk
from nltk.stem import WordNetLemmatizer
import string

def preprocess_data(data,data_X,data_y):
    words = []
    classes = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)  # tokenize each pattern 
            words.extend(tokens)  # and append tokens to words 
            # appending pattern to data_X
            data_X.append(pattern)
            # appending the associated tag to each pattern
            data_y.append(intent["tag"])

        # adding the tag to the classes if it's not there already
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

    # initializing lemmatizer to get stem of words
    lemmatizer = WordNetLemmatizer()
    # lemmatize all the words in the vocab and convert them to lowercase
    # if the words don't appear in punctuation
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
    # sorting the vocab and classes in alphabetical order and taking the
    # set to ensure no duplicates occur
    words = sorted(set(words))
    classes = sorted(set(classes))
    
    return words, classes, data_X, data_y

# # Example usage:
# data = {
#     "intents": [
#         {"tag": "greeting", "patterns": ["Hi", "Hello"]},
#         {"tag": "farewell", "patterns": ["Goodbye", "See you later"]}
#     ]
# }

# print("Words:", words)
# print("Classes:", classes)

def clean_text(text):
    # Tokenizing the text into words
    tokens = nltk.word_tokenize(text)
    
    # Lemmatizing each word to its base form
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens

# Function to create a bag of words representation of a text
def bag_of_words(text, vocab):
    # Tokenize the text
    tokens = clean_text(text)
    
    # Initialize bag of words vector with zeros
    bow = [0] * len(vocab)
    
    # Iterate over each token
    for token in tokens:
        # Iterate over each word in the vocabulary
        for idx, word in enumerate(vocab):
            # If the token matches a word in the vocabulary, set the corresponding index in bag of words vector to 1
            if word == token:
                bow[idx] = 1
    
    # Convert the bag of words vector to numpy array
    return np.array(bow)

def pred_class(text, vocab, labels, model):
    # Convert the input text into a bag-of-words representation using the provided vocabulary
    bow = bag_of_words(text, vocab)
    
    # Use the model to predict the probabilities of each class/tag
    # result = model.predict(np.array([bow]))[0]  # Extracting probabilities
    # Assuming `model` is an instance of MyClassModel
    # `text` is the input text and `vocab` is the vocabulary
    result = custom_predict(model, text, vocab)

    # Define a threshold to filter out low-confidence predictions
    thresh = 0.5
    
    # Filter out predictions with probability greater than the threshold
    y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
    
    # Sort the filtered predictions by their probability in decreasing order
    y_pred.sort(key=lambda x: x[1], reverse=True)  # Sorting by values of probability in decreasing order
    
    # Initialize an empty list to store the predicted labels
    return_list = []
    
    # Iterate over the filtered and sorted predictions
    for r in y_pred:
        # Append the corresponding label (tag) to the return list
        return_list.append(labels[r[0]])  # Contains labels (tags) for highest probability
    
    # Return the list of predicted labels
    return return_list

def custom_predict(model, text, vocab):
        # Convert the input text into a bag-of-words representation using the provided vocabulary
        bow = bag_of_words(text, vocab)
        # Make predictions using the model
        result = model.predict(np.array([bow]),verbose=None)[0]  # Extracting probabilities
        return result


