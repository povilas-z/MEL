# MEL

*Web app for Melanoma detection using PyTorch & Flask*

## Introduction

*Disclaimer: MEL is not designed to be a substitute for medical advice. If you are concerned about something see a trained medical professional.*

MEL is the intersection of my fascination with skincare and machine learning. In exploring these interests, I aimed to create a complete and accessible project that could make a meaningful difference.

MEL seeks to elevate the importance of self-care and awareness, encouraging individuals to take action if they have concerns about their skin health.

## Tech Stack

* React (FrontEnd) 
* NEXT JS (Styling)
* PyTorch (Model)

## Model Creation

* ResNet50 as base model
* HAM10000 is used as training, and test data set
* Layers are subsequently unfrozen depending on loss stalling (learning rate adjusted accordingly)
* Model with lowest loss is saved

## Future Steps

### User interface

* Accessible user interface where users are able to access MEL's model and upload individual pictures
* User pictures will be returned with a guess of skin lesion and a confidence value
