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
import numpy as np
from PIL import Image

def predict_image(model, image_path):
    target_dimensions = (64, 64)
    img = Image.open(image_path)
    img = img.resize(target_dimensions)
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = img_array.reshape(1, 64, 64, 3)
    prediction = model.predict(img_array)
    return prediction

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
            prediction = predict(self,path='A_Test.jpg')
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



