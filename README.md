# CodeForLife-Chatbot
Chatbot designed using torch.nn and used flask for web development 


Running the chatbot:

#clone this repo:
$ git clone https://github.com/Sachdevah/CodeForLife-Chatbot.git
$ cd CodeForLife-Chatbot

#create and activate a virtual environment called "venv"
$ python3 -m venv venv
$ . venv/bin/activate


After activating venv, Installing dependencies:
$ (venv) pip install Flask torch torchvision nltk

Install nltk package:
$ (venv) python
>>> import nltk
>>> nltk.download('punkt')

Training the chatbot:
NOTE: after training it creates a file of trained data called "data.pth" as I have already trained the data before and also included this "data.pth" file, you can skip this file unless you made more changes to the intents.json. In that case, you need to train the model again.
$ (venv) python train.py

finally to run the flask project, run chat.py file
$ (venv) python chat.py
