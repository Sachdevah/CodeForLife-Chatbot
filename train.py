import numpy as np
import random
import json
from model import NeuralNet
import torch
from nltk_utils import bag_of_words, tokenize,stem
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#opening the intents.json on read mode
with open('intents.json', 'r') as Intentfile:
    #takes a file object and returns a json object in a variable intents
    intents = json.load(Intentfile)

# """
# all_words= to store all the tokenised words from intents.json
# tags= to store all tags from intents.json
# xy= to store key-value pairs of both 
# """
all_words = []
tags = []
xy = []

# """
# loop through each sentence in our patterns in intents.json

# """
for intent in intents['intents']:
    tag = intent['tag']
    #adding tags from the intents.json to tags array
    tags.append(tag)
    for pattern in intent['patterns']:
        #tokenising all the words in the intents file or separating all the words
        w = tokenize(pattern)
        #adding all the collected words in a all_words array
        all_words.extend(w)
        #adding xy pairs in xy array where xy represents key-value pairs 
        xy.append((w, tag))

# remove puctuations from all_words
ignore_words = ['?', '.', '!']
# """
# #stem all words
# #removes duplicates and sorts the array
# """
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))


#printing all patterns, tags and all_words
print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)


# """
# creating training data, xy key-value pairs
# X reprensents bag of words from all the input sentences
# Y is the tags from json file
# """
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

# """
# saving trained XY
# """
X_train = np.array(X_train)
y_train = np.array(y_train)
#number of times you want to run the training cycle for our case it's 1300/100=13 cycles
NumberOfEpochs = 1000
batch_size = 8
learning_rate = 0.001
#inputing bag_of_words
input_size = len(X_train[0])
#can change hidden_side as desired
hidden_size = 8
#output_size represents the number of different tags available in json
output_size = len(tags)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # **support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # **we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
#num_worker represents threads to do the task 
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

#check if gpu available, if not then use cpu 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#creating an instance of our neural net NN_Model
NN_Model = NeuralNet(input_size, hidden_size, output_size).to(device)

#check Loss and optimizer for our NN_Model
entropyLoss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(NN_Model.parameters(), lr=learning_rate)

#Training the NN_Model
for epoch in range(NumberOfEpochs):

    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # moving forward in layers
        outputs = NN_Model(words)
        # **if y would be one-hot, we must apply
        dataLoss = entropyLoss(outputs, labels)
        
        #Backward pass in layer and optimize the output
        optimizer.zero_grad()
        dataLoss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{NumberOfEpochs}], Data Loss: {dataLoss.item():.5f}')


print(f'final Data loss: {dataLoss.item():.5f}')
#storing data 
data = {
"model_state": NN_Model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

#saving data in a file called trainedData.pth, if it exists then it updates the file otherwise creates a new one
FILE = "trainedData.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
