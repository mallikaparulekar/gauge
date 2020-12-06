
import cv2
from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import numpy as np

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)
set_session(session)


class FacialExpressionModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        #self.loaded_model.compile()
        #self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        global session
        set_session(session)
        self.preds = self.loaded_model.predict(img)

        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]





#img = cv2.imread("/Users/sharad/Downloads/confused/image.jpg",0)

#img = cv2.imread("/Users/sharad/PycharmProjects/HackDukeML/venv/demo/confused.jpg",0)
img = cv2.imread("/Users/sharad/PycharmProjects/HackDukeML/venv/demo/not_confused.jpg",0)

#img = cv2.resize(img, (48, 48))

model = FacialExpressionModel("/Users/sharad/PycharmProjects/HackDukeML/venv/model.json", "/Users/sharad/PycharmProjects/HackDukeML/venv/model_weights.h5")

#model.predict_emotion(img)
emotion = model.predict_emotion(img[np.newaxis, :, :, np.newaxis])

if(emotion == "Disgust" or emotion == "Angry"):
    print("The student is confused -- notifying the teacher")
    print("The original emotion was" , emotion)

else:
    print("all good! No confusion")
    print("The original emotion was", emotion)

#issue was put in same directpry. Have to resize for now! color is fine.
