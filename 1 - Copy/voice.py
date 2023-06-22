# %%
# Load EDA Pkgs
import pandas as pd
import numpy as np

# %%
# Load Data Viz Pkgs
import seaborn as sns

# %%
# Load Text Cleaning Pkgs
import neattext.functions as nfx

# %%
# Load ML Pkgs
# Estimators
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# %%
# Load Dataset
df = pd.read_csv("emotion_dataset_raw.csv")

# %%
df.head()

# %%
# Value Counts
df['Emotion'].value_counts()

# %%
# Plot
sns.countplot(x='Emotion',data=df)

# %%
# Data Cleaning
dir(nfx)

# %%
# User handles
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)

# %%
# Stopwords
df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)

# %%
df

# %%
# Features & Labels
Xfeatures = df['Clean_Text']
ylabels = df['Emotion']

# %%
#  Split Data
x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=0.3,random_state=42)

# %%
# Build Pipeline
from sklearn.pipeline import Pipeline

# %%
# LogisticRegression Pipeline
pipe_lr = Pipeline(steps=[('cv',CountVectorizer()),('lr',LogisticRegression())])

# %%
# Train and Fit Data
pipe_lr.fit(x_train,y_train)

# %%
pipe_lr

# %%
# Check Accuracy
pipe_lr.score(x_test,y_test)

# %%
import speech_recognition as sr
recognizer = sr.Recognizer()
''' recording the sound '''

with sr.Microphone() as source:
    print("Adjusting noise ")
    recognizer.adjust_for_ambient_noise(source, duration=1)
    print("Recording for 5 seconds")
    recorded_audio = recognizer.listen(source, timeout=5)
    print("Done recording")

''' Recorgnizing the Audio '''
try:
    print("Recognizing the text")
    text = recognizer.recognize_google(
            recorded_audio, 
            language="en-US"
        )
    print("Decoded Text : {}".format(text))

except Exception as ex:
    print(ex)

#predicted_array = pipe_lr.predict([text])

# %%
# Prediction Prob
pipe_lr.predict_proba([text])

# %%
# To Know the classes
pipe_lr.classes_

# %%
# Save Model & Pipeline
import joblib

pipeline_file = open("voice.pkl","wb")
joblib.dump(pipe_lr,pipeline_file)
pipeline_file.close()

# %%
# Load the saved model
import pickle
with open('voice.pkl', 'rb') as file:
    model = pickle.load(file)

# %%
predicted_array = pipe_lr.predict([text])

# %%
print(predicted_array)
import tkinter as tk
import time

class ProgressBar(tk.Frame):
    def __init__(self, master=None, maximum=100, **kwargs):
        super().__init__(master, **kwargs)
        self.maximum = maximum
        self.value = 0
        
        self.progressbar = tk.Canvas(self, bg="white", width=200, height=20)
        self.progressbar.pack()
        self.progressbar.create_rectangle(0, 0, 0, 20, fill="green", outline="")
        
    def update(self, value):
        self.value = value
        if self.value > self.maximum:
            self.value = self.maximum
        progress_width = 200 * self.value / self.maximum
        self.progressbar.delete("all")
        if self.value <= self.maximum/2:
            self.progressbar.create_rectangle(0, 0, progress_width, 20, fill="green", outline="")
        elif self.value <= 3*self.maximum/4:
            self.progressbar.create_rectangle(0, 0, progress_width, 20, fill="yellow", outline="")
        else:
            self.progressbar.create_rectangle(0, 0, progress_width, 20, fill="red", outline="")
        self.progressbar.update()
        
    def start(self):
        i = 0
        while i <= self.maximum:
            self.update(i)
            time.sleep(0.05)
            if predicted_array=='joy' and i==14:
                break
            elif predicted_array=='surprise' and i==29:
                break
            elif predicted_array=='fear' and i==43:
                break
            elif predicted_array=='neutral' and i==57:
                break
            elif predicted_array=='disgust' and i==71:
                break
            elif predicted_array=='anger' and i==86:
                break
            elif predicted_array=='sadness' and i==100:
                break
            i += 1

                    
root = tk.Tk()
root.geometry("300x100")

progressbar = ProgressBar(root, maximum=100)
progressbar.pack(pady=10)

progressbar.start()

root.mainloop()
