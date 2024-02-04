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
import mediapipe as mp

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connection

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=0),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=0)
                             )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=0),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=0)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=0),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=0)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=0),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=0)
                             )
    
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Populate folders:

# # Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data3')

# # Actions that we try to detect
actions = np.array(['hello','how','you'])

no_sequences = 30

# Videos are going to be 60 frames in length, approx. 2 sec
sequence_length = 30

# Folder start
start_folder = 30

colors = [(245,117,16), (117,245,16), (16,117,245),(245,117,16), (117,245,16), (16,117,245),(16,117,245),(16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -2)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


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

    def update(self):
        global Generated_text
        return_value, frame = self.capture.read()
        if return_value:
            texture = self.texture
            w, h = frame.shape[1], frame.shape[0]
            if not texture or texture.width != w or texture.height != h:
                self.texture = texture = Texture.create(size=(w, h))
                texture.flip_vertical()
                texture.flip_horizontal()
            # 1. New detection variables
            sequence = []
            sentence = []
            threshold = 0.8

            # Set mediapipe model 
            with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                print(results)
                
                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # 2. Prediction logic
                keypoints = extract_keypoints(results)
        #         sequence.insert(0,keypoints)
        #         sequence = sequence[:30]
                sequence.append(keypoints)
                sequence = sequence[-30:]
                
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    
                    
                #3. Viz logic
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]

                    # Viz probabilities
                    image = prob_viz(res, actions, image, colors)
                    
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Show to screen
                # cv2.imshow('OpenCV Feed', image)
                # texture.blit_buffer(image.tobytes(),)
            texture.blit_buffer(image.tobytes(), colorfmt='bgr')
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



