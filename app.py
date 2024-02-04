from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics.texture import Texture
from kivy.base import EventLoop
from kivy.uix.scrollview import ScrollView
from kivy.properties import StringProperty
import cv2
import datetime
import time
import numpy as np
import os
import tensorflow as tf

model =tf.keras.models.load_model('ASL2.h5')
# Prediction function
def predict(self, *args, path):
    global model
    # Specify thresholds
    detection_threshold = 0.95
    arr = cv2.imread(path)
    new_array = cv2.resize(arr,(32,32))
    new_arr = new_array.reshape(1,32,32,-1)
    predictions = model.predict(new_arr) ##predict classes
    predictionsss = model.predict(new_arr)[0]
    predicti = np.argmax(predictions, axis=1) ##get the index of the highest probability
    
    # Detection Threshold: Metric above which a prediciton is considered positive 
    detection = np.sum(np.array(predicti[0]) > detection_threshold)

    if detection:
        return predictionsss[2]

def TextWithTime(oldtext, newtext):
    date = str(datetime.date.today())
    local_time =str( time.strftime("%I:%M:%S"))
    return oldtext + "\n\n" + "[ " + date + " "+ local_time + " ]: " + newtext

capture = None
Generated_text = StringProperty()
Generated_text = ""


class SecondWin(Screen):

    def on_enter(self):
        global Generated_text

        self.manager.text_gen = Generated_text
        self.manager.text_gen = TextWithTime(self.manager.text_gen,"abdhssss")




class WinManager(ScreenManager):
    global Generated_text
    text_gen =StringProperty("1234")
    



class KivyCamera(Image):

    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = None

    def start(self, capture, fps=1):
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def stop(self):

        Clock.unschedule_interval(self.update)
        self.capture = None

    def update(self, dt):
        global Generated_text
        return_value, frame = self.capture.read()
        if return_value:
            texture = self.texture
            w, h = frame.shape[1], frame.shape[0]
            if not texture or texture.width != w or texture.height != h:
                self.texture = texture = Texture.create(size=(w, h))
                texture.flip_vertical()
                texture.flip_horizontal()
            predict_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            predict_image = cv2.resize(predict_image,(32,32))
            cv2.imwrite('prediction.png',predict_image)
            prediction = predict(self,path='prediction.png')
            print(prediction)
            texture.blit_buffer(frame.tobytes(), colorfmt='bgr')
            self.canvas.ask_update()





class CamHome(Screen):

    def init_cam(self):
        pass

    def dostart(self, *largs):
        global capture
        capture = cv2.VideoCapture(0)
        self.ids.cam.start(capture)

    def doexit(self):
        global capture

        if capture != None:
            capture.release()
            capture = None
        # EventLoop.close()




kv = Builder.load_file("style.kv")


class MyMainApp(App):
    
    def build(self):
        global model
        model = tf.keras.models.load_model('ASL2.h5')
        return kv


if __name__ == "__main__":
    MyMainApp().run()



