import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import gc
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
import cv2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import skimage
from skimage.transform import resize
import tensorflow as tf
from tensorflow import keras
import os

##init params
batch_size = 16 ##number of pics per patch to be taken
image_size = 32 ##image size
target_dimensions = (image_size, image_size, 3) ##64x64 RGB
num_classes = 29 ##number of classes
epochs = 50 ##number of epochs

##num of train pics, and directory
train_len = 87000
train_dir = '/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train'

##map each folder with pictures to a number
def get_data(folder):
    X = np.empty((train_len, image_size, image_size, 3), dtype=np.float32)
    y = np.empty((train_len,), dtype=np.int64) 
    cnt = 0 ##counter
    label_mapping = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
        'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
        'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26,
        'nothing': 27, 'space': 28
    } ##dictionary of labels
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            label = label_mapping.get(folderName, 29)
            for image_filename in os.listdir(folder + '/' + folderName): ##loop through each image in the folder
                img_file = cv2.imread(folder  + '/' + folderName + '/' + image_filename) ##read image
                if img_file is not None: ##if image is not empty
                    img_file = skimage.transform.resize(img_file, (image_size, image_size, 3)) ##resize image to 64x64x3
                    img_arr = np.asarray(img_file).reshape((-1, image_size, image_size, 3)) ##reshape image to 64x64x3
                    
                    X[cnt] = img_arr ##add image to X
                    y[cnt] = label ##add label to y
                    cnt += 1 
    print('done loading')
    return X, y 

##split data into train and test
def splitting(X_train,y_train):
    X_data = X_train 
    y_data = y_train

    ##70/30 split of train/test, need test for cross validation
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3,random_state=42,stratify=y_data)

    #one hot encoding
    y_cat_train = to_categorical(y_train,29) ##29 classes, converts training labels to something the model can take
    y_cat_test = to_categorical(y_test,29) ##same hena bs l test labels

    ##don't need this anymore, we only use x_train and y_train so delete this to save memory
    del X_data 
    del y_data
    print('splitting complete')
    return X_train, X_test, y_train, y_test, y_cat_train, y_cat_test

##start building the sequential model and training
def modelbuilding(X_train, y_cat_train, X_test, y_cat_test):
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=target_dimensions)) ##32 filters, 5x5 kernel, input shape is 64x64x3
    model.add(Activation('relu')) ##activation function
    model.add(MaxPooling2D((2, 2))) ##pooling layer
    model.add(Conv2D(64, (3, 3))) ##64 filters, 3x3 kernel
    model.add(Activation('relu')) ##activation function
    model.add(MaxPooling2D((2, 2))) ##pooling layer
    model.add(Conv2D(64, (3, 3))) ##64 filters, 3x3 kernel
    model.add(Activation('relu')) ##activation function
    model.add(MaxPooling2D((2, 2))) ##pooling layer
    model.add(Flatten()) ##flatten the data
    model.add(Dense(128, activation='relu')) ##128 neurons, activation function
    model.add(Dense(29, activation='softmax')) ##output layer, 29 neurons, activation function

    model.summary()

    early_stop = EarlyStopping(monitor='val_loss',patience=4) ##stop training if val_loss doesn't improve after 4 epochs
    checkpoint_filepath = os.path.join('checkpoints', 'op' + \
        '.{epoch:03d}-{val_loss:.3f}-best.h5'),
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        verbose=1,
        mode='max',
        save_best_only=True)

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']) ##compile model, adam optimizer, loss function, accuracy metric

    ##start training el model
    model.fit(X_train, y_cat_train,
            epochs=epochs,
            batch_size=batch_size, 
            verbose=2, #wanna keep this at 2 to see progress, keda keda early stopping is gonna stop for me
            validation_data=(X_test, y_cat_test), ##allows for cross validation 
            # callbacks=[early_stop] ##early stopping
            # callbacks=[model_checkpoint_callback] ##save best weights
    )

    metrics = pd.DataFrame(model.history.history) ##save history of model to a dataframe
    print('done training')
    model.save('ASL2.h5') ##save weights
    return metrics,model,X_test,y_test,y_cat_test

def eval(metrics,model, X_test, y_test, y_cat_test):
    metrics[['loss','val_loss']].plot() ##plot loss and val_loss as per requested

    metrics[['accuracy','val_accuracy']].plot() ##plot loss and val_loss as per requested

    model.evaluate(X_test,y_cat_test,verbose=0) ##evaluate model

    predictions = model.predict(X_test) ##predict classes for test data to be able to see confusion matrix
    predictions = np.argmax(predictions, axis=1) ##get the index of the highest probability

    print(classification_report(y_test,predictions))

    plt.figure(figsize=(12,12)) ##plot confusion matrix
    sns.heatmap(confusion_matrix(y_test,predictions)) ##plot confusion matrix
    plt.show()

if __name__ == "__main__":
    X_train, y_train = get_data(train_dir) ##get train data
    X_train, X_test, y_train, y_test, y_cat_train, y_cat_test = splitting(X_train,y_train) ##split data
    metrics,model,X_test,y_test,y_cat_test = modelbuilding(X_train, y_cat_train, X_test, y_cat_test) ##build model and train
    eval(metrics,model, X_test, y_test, y_cat_test) ##evaluate model