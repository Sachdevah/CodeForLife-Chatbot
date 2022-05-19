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

Image of the interface:
<img width="425" alt="Screenshot 2022-05-19 at 11 38 01" src="https://user-images.githubusercontent.com/70515316/169274693-daa06fc4-57c6-4c94-bfac-f1a057ca92a9.png">
