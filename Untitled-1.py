
# Iterating over all the intents
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens= nltk.word_tokenize(pattern) # tokenize each pattern 
        words.extend(tokens) #and append tokens to words 
        data_X.append(pattern) #appending pattern to data_X
        data_y.append(intent["tag"]),# appending the associated tag to each pattern
    
    # adding the tag to the classes if it's not there already
    if intent["tag"] not in classes:
        classes.append(intent["tag"])
        
#initializing lemmatizer to get stem of words.
lemmatizer= WordNetLemmatizer()
#lemmatize all the words in the vocab and convert them to lowercase
# if the words don't appear in punctuation
words= [lemmatizer.lemmatize (word.lower()) for word in words if word not in string.punctuation]
# sorting the vocab and classes in alphabetical order and taking the # set to ensure no duplicates occur words sorted(set(words))
classes= sorted(set(classes))