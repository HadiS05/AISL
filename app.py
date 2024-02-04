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




def TextWithTime(oldtext, newtext):
    date = str(datetime.date.today())
    local_time =str( time.strftime("%I:%M:%S"))
    return oldtext + "\n\n" + "[ " + date + " "+ local_time + " ]: " + newtext

capture = None
Generated_text = StringProperty()
Generated_text = "sfeskfeksjfnekjfnskfnsfnes;ef;eisjf;osijfsfsuhusnvsuhefs"*90


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

    def start(self, capture, fps=30):
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def stop(self):

        Clock.unschedule_interval(self.update)
        self.capture = None

    def update(self, dt):
        return_value, frame = self.capture.read()
        if return_value:
            texture = self.texture
            w, h = frame.shape[1], frame.shape[0]
            if not texture or texture.width != w or texture.height != h:
                self.texture = texture = Texture.create(size=(w, h))
                texture.flip_vertical()
                texture.flip_horizontal()
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
        return kv


if __name__ == "__main__":
    MyMainApp().run()



