import random
import json
# py torch, tensorflow, keras that can be used for this purpose
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize



#open intents.json file on read mode 
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
#loading the trained data NN_Model

#FILE = "trainedData.pth"
#loading the trained data stored in trainedData.pth
data = torch.load("trainedData.pth")

#assigning the variables to the collected data
input_size = data["input_size"]
hidden_size = data["hidden_size"]
all_words = data['all_words']
tags = data['tags']
output_size = data["output_size"]
model_state = data["model_state"]


#check if gpu available, otherwise use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#using our NN NN_Model to load a device then using eval() on that device
NN_Model = NeuralNet(input_size, hidden_size, output_size).to(device)
#Loads a model's parameter dictionary using a deserialized state_dict
NN_Model.load_state_dict(model_state)
#to evaluate model and deactivate layer while doing that
NN_Model.eval()


#function to predict the user intent and respond accordingly
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    #reshape method from numpy
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = NN_Model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    #an activation function for output layer, predict a multinomial probability distribution
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "Sorry, I didn't understand..."


#imports to run flask app
from flask import Flask, render_template, request,jsonify
#from chat import get_response
app=Flask(__name__)

#syntax new to flask 2.0--or use @app.route("/",methods=["GET"])
@app.get("/")
def index_get():
    return render_template("/base.html")
     
@app.post("/predict")
def predict():
    text=request.get_json().get("message")
    # TODO: check if text is valid
    #using get_response function 
    response=get_response(text)
    message={"answer": response}
    return jsonify(message)

if __name__ =="__main__":
    app.run(debug=True)
    
    