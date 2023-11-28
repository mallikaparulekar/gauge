# gauge
An app for teachers to gauge student understanding - Best Education Hack at HackDuke 2020

This repo contains code for a convoluted neural network (CNN) that detects the seven universal facial expressions. [Anger, Disgust, Fear, Happiness, Neutral, Sadness, Fear].  

This neural network plays an essential role in Gauge. Gauge is a zoom add-on that allows teachers to better understand how their students are learning the content being taught. The app uses this facial expression model to determine if the students are confused/don’t understand the content and based on that, alert the teacher. 

Facial_Expression_Training.py contains code that creates multiple layers of the neural network and trains it on of images of all kinds of facial expressions.

Camera.py contains code for analyzing expressions in video feeds using a facial detection model that puts a bounding box around a person’s face. 

Model.py uses the trained model along with its weights to develop predictions when given an image/video feed.

The test folder contains images used for validation and testing

Credit to https://www.coursera.org/projects/facial-expression-recognition-keras for the tutorial.

