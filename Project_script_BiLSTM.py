# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 00:08:44 2023

@author: Archana
"""

import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa
import librosa.display
import librosa.util
from IPython.display import Audio
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import keras.utils
from keras import layers
from keras import models
from keras import optimizers
import tensorflow as tf
import pandas as pd

# Write the classified data into .wav file
import soundfile as sf
df1=pd.read_csv('S1_py.csv')
df2=pd.read_csv('S2_py.csv')
df3=pd.read_csv('S3_py.csv')
df4=pd.read_csv('S4_py.csv')

samplerate=16000
sf.write("S1_py.wav", np.ravel(df1), samplerate)
sf.write("S2_py.wav", np.ravel(df2), samplerate)
sf.write("S3_py.wav", np.ravel(df3), samplerate)
sf.write("S4_py.wav", np.ravel(df4), samplerate)

# 3 classes : 0 S1, 1 S2, 2 systole or Diastole
class_names = ['S1', 'S2', 'Systole','Diastole']
labels = {'S1' : 0, 'S2' : 1, 'Systole' : 2, 'Diastole':3}

x = []
y = class_names #or labels

# @note: resampling can take some time!
# signal, _ = librosa.load('./Train_data/S1.wav', sr=8000, mono=True, duration=30, dtype=np.float32)
for i in range(4):
    file_path = os.path.join('S'+str(i+1)+'_py.wav')
    # @note: resampling can take some time!
    signal, _ = librosa.load(file_path, sr=16000, mono=True, duration=40, dtype=np.float32)
    x.append(signal)
    
plt.figure(figsize=(12, 2))
librosa.display.waveshow(x[3], sr=16000)
plt.title(y[3])
plt.show()

Audio(x[0], rate=16000)

# sudo apt-get install python3-tk
x_framed = []
y_framed = []

#change frame_length to 16000 hop_length to 512
for i in range(len(x)):
    frames = librosa.util.frame(x[i], frame_length=16896, hop_length=512)
    x_framed.append(np.transpose(frames))
    y_framed.append(np.full(frames.shape[1], y[i]))
    
# merge sliced frames and label
x_framed = np.asarray(x_framed)
y_framed = np.asarray(y_framed)
x_framed = x_framed.reshape(x_framed.shape[0]*x_framed.shape[1], x_framed.shape[2])
y_framed = y_framed.reshape(y_framed.shape[0]*y_framed.shape[1], )

print("x_framed shape: ", x_framed.shape) # Each frame can be used to create spectrogram 
print("y_framed shape: ", y_framed.shape) # Corresponding label for each frame

x_features = []
y_features = y_framed

for frame in tqdm(x_framed):
    # Create a mel-scaled spectrogram change sr value to 16000
    S_mel = librosa.feature.melspectrogram(y=frame, sr=16000, n_mels=30, n_fft=1024, hop_length=512, center=False)
    # Scale according to reference power
    S_mel = S_mel / S_mel.max()
    # Convert to dB
    S_log_mel = librosa.power_to_db(S_mel, top_db=80.0)
    x_features.append(S_log_mel)

# Convert into numpy array
x_features = np.asarray(x_features)
print(x_features.shape)
print(y_framed)

# Flatten features for scaling
x_features_r = np.reshape(x_features, (len(x_features), 30*32))

# Create a feature scaler
scaler = preprocessing.StandardScaler().fit(x_features_r)

# Apply the feature scaler 
x_features_s = scaler.transform(x_features_r)

# Convert labels to categorical one-hot encoding
from sklearn.preprocessing import LabelEncoder
import numpy as np

code = y_features

label_encoder = LabelEncoder()
vec = label_encoder.fit_transform(code)
y_features_cat = keras.utils.to_categorical(vec, num_classes=len(class_names))

print("y_features_cat shape: " ,y_features_cat.shape)

x_train, x_test, y_train, y_test = train_test_split(x_features_s,
                                                    y_features_cat,
                                                    test_size=0.25,
                                                    random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=0.25,
                                                  random_state=1)

print('Training samples:', x_train.shape)
print('Validation samples:', x_val.shape)
print('Test samples:', x_test.shape)


from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM

# Building the BiLSTM model
model = models.Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(30, 32)))
model.add(Bidirectional(LSTM(50)))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(9, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))
# print model summary
model.summary()

# Choose the Adam optimizer and learning rate
adam = optimizers.Adam(learning_rate = 0.01)

# Compile the model and choose the evaluation metrics
model.compile(optimizer=adam, loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Reshape features to include channel
x_train_r = x_train.reshape(x_train.shape[0], 30, 32, 1)
x_val_r = x_val.reshape(x_val.shape[0], 30, 32, 1)
x_test_r = x_test.reshape(x_test.shape[0], 30, 32, 1)


history = model.fit(x_train_r, y_train, validation_data=(x_val_r, y_val),
                    batch_size=500, epochs=50, verbose=2)


train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.clf()
plt.xlabel('Epochs', fontweight='bold')  # Adding bold fontweight to x-axis label
plt.ylabel('Loss', fontweight='bold')    # Adding bold fontweight to y-axis label
plt.plot(train_loss, color='tab:blue', label='Training Loss', linewidth=2)  # Increase curve thickness using linewidth
plt.plot(val_loss, color='chocolate', label='Validation Loss', linewidth=2)  # Increase curve thickness using linewidth
plt.title('a). BiLSTM-4 class model', fontsize=10, fontweight='bold')  # Adding bold fontweight to title
plt.legend()

# Increase linewidth of x-axis and y-axis
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)

# Increase tick width of x-axis and y-axis
plt.tick_params(width=1.5)
plt.savefig('Loss_epoch_lstm_4class_2.png', dpi=400)

# Evaluate Accuracy
# Next, compare how the model performs on the test dataset:
print('Evaluate model:')
results = model.evaluate(x_test_r, y_test)
test_loss, test_accuracy = model.evaluate(x_test_r, y_test)

# print('Test Loss:', test_loss)
# print('Test Accuracy:', test_accuracy)

print(results)
print('Test loss: {:f}'.format(results[0]))
print('Test accuracy: {:.2f}%'.format(results[1] * 100))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred = model.predict(x_test_r)

y_pred_class_nb = np.argmax(y_pred, axis=1)
y_true_class_nb = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true_class_nb, y_pred_class_nb)
np.set_printoptions(precision=2)
print("Accuracy = {:.2f}%".format(accuracy * 100))

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

# Assuming you already have the true labels y_true and predicted labels y_pred
y_pred = model.predict(x_test_r)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

# Calculate F1 score
f1 = f1_score(np.argmax(y_test, axis=1), y_pred_classes, average='weighted')

# Calculate precision
precision = precision_score(np.argmax(y_test, axis=1), y_pred_classes, average='weighted')

# Calculate recall
recall = recall_score(np.argmax(y_test, axis=1), y_pred_classes, average='weighted')

# Print the results
print('F1 score: {:.2f}'.format(f1))
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))

classes = class_names
cm = confusion_matrix(y_true_class_nb, y_pred_class_nb, labels=[0,1,2,3])

# (optional) normalize to get values in %
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

# We want to show all ticks...
ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       ylabel='True labels', xlabel='Predicted labels')

# Loop over data dimensions and create text annotations.
thresh = cm.max() / 2.

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")


plt.title('a). BiLSTM-4 class', fontsize=11)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.savefig('confusion_lstm_4class_2.png', dpi=400)

#%%
import numpy as np
import matplotlib.pyplot as plt

# Sample model output for a sequence of length 100, with 4 classes (for illustration purposes)
num_classes = 4
model_output = y_pred


# Sliding window parameters
window_size = 10
stride = 5

# Function to apply sliding window and aggregate predictions
def apply_sliding_window(predictions, window_size):
    window_predictions = []
    for i in range(0, len(predictions) - window_size + 1):
        window = predictions[i:i + window_size]
        aggregated_prediction = np.mean(window, axis=0)  # Example: Using mean for aggregation
        window_predictions.append(aggregated_prediction)
    return np.array(window_predictions)

# Applying sliding window on the model output
window_predictions = apply_sliding_window(model_output, window_size, stride)

# Plotting the output
plt.figure(figsize=(10, 6))
plt.title("Sliding Window Output")
plt.xlabel("Time Step")
plt.ylabel("Class Probability")
plt.grid(True)

for class_idx in range(num_classes):
    plt.plot(window_predictions[:, class_idx], label=f"Class {class_idx}")

plt.legend()
plt.show()


import numpy as np

# Sample model output for 5 instances with 4 classes (for illustration purposes)

num_classes = 4
model_output = y_pred

# Function to apply maximum voting
def apply_maximum_voting(predictions):
    return np.argmax(predictions, axis=1)

# Applying maximum voting to get the predicted class indices
predicted_class_indices = apply_maximum_voting(model_output)

# Assuming class labels as strings (e.g., "Class 0", "Class 1", "Class 2", "Class 3")
class_labels = ["1", "3", "2", "4"]

# Mapping predicted class indices to class labels
predicted_class_labels = [class_labels[idx] for idx in predicted_class_indices]

# Print the results
print("Model Output:")
print(model_output)
print("\nPredicted Class Indices:")
print(predicted_class_indices)
print("\nPredicted Class Labels:")
print(predicted_class_labels)


def apply_maximum_voting(predictions):
    return np.argmax(predictions, axis=1)

# Applying maximum voting to get the predicted class indices
predicted_class_indices = apply_maximum_voting(model_output)

# Assuming class labels as strings (e.g., "Class 0", "Class 1", "Class 2", "Class 3")
class_labels = ["1", "3", "2", "4"]

# Mapping predicted class indices to class labels
predicted_class_labels = [class_labels[idx] for idx in predicted_class_indices]

# Plotting the output in a curve form
plt.figure(figsize=(8, 6))
plt.plot(range(model_output), predicted_class_indices, marker='o', linestyle='-', color='b')
plt.xticks(range(model_output), [f"Instance {i}" for i in range(model_output)], rotation=45)
plt.xlabel("Instances")
plt.ylabel("Predicted Class Indices")
plt.title("Maximum Voting Output")
plt.grid(True)
plt.show()

#%%

import numpy as np

# Suppose 'predict_prob' contains the probabilities predicted by the model

# Define the length of the sliding window
window_length = 2

# Create an empty array to store the final classifications
final_classifications = []

# Apply the sliding window to each prediction time point
for i in range(len(y_pred)):
    # Calculate the starting and ending indices for the sliding window
    start_index = max(0, i - window_length // 2)
    end_index = min(len(y_pred), i + window_length // 2)
    
    # Extract the window of probabilities
    window = y_pred[start_index:end_index]
    
    # Assign the final classification for the current time point
    final_classifications.append(np.argmax(y_pred[i]))

print(final_classifications)
x = final_classifications

plt.plot(x[0:20])
