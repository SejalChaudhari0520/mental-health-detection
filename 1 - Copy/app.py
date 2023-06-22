
import nltk
from sklearn import preprocessing
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import tensorflow as tf
from keras.models import load_model

model = load_model('model.h5')
import json
import random
from flask import Flask, jsonify,render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
app = Flask(__name__)
app.secret_key = 'xyzsdfg'
  
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'user_system'
  
mysql = MySQL(app)
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

@app.route('/')
@app.route('/login', methods =['GET', 'POST'])
def login():
    mesage = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s AND password = % s', (email, password, ))
        user = cursor.fetchone()
        if user:
            session['loggedin'] = True
            session['userid'] = user['userid']
            session['name'] = user['name']
            session['email'] = user['email']
            mesage = 'Logged in successfully !'
            return render_template('user.html', mesage = mesage)
        else:
            mesage = 'Please enter correct email / password !'
    return render_template('login.html', mesage = mesage)
  
@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('userid', None)
    session.pop('email', None)
    return redirect(url_for('login'))
  
@app.route('/register', methods =['GET', 'POST'])
def register():
    mesage = ''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form :
        userName = request.form['name']
        password = request.form['password']
        email = request.form['email']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = % s', (email, ))
        account = cursor.fetchone()
        if account:
            mesage = 'Account already exists !'
            return render_template('register.html', mesage = mesage)
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            mesage = 'Invalid email address !'
            return render_template('register.html', mesage = mesage)
        elif not userName or not password or not email:
            mesage = 'Please fill out the form !'
            return render_template('register.html', mesage = mesage)
        else:
            cursor.execute('INSERT INTO user VALUES ( Null,% s, % s, % s)', (userName, email, password, ))
            mysql.connection.commit()
            mesage = 'You have successfully registered !'
            return render_template('login.html', mesage = mesage)
    elif request.method == 'POST':
        mesage = 'Please fill out the form !'
    return render_template('register.html', mesage = mesage)




from flask import Flask, render_template, request

app.static_folder = 'static'

@app.route("/home")
def home():
    return render_template("user.html")

@app.route("/chatbot")

def chatbot():
    return render_template("chatbot.html")

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=True)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res



@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

@app.route("/user")
def user():
    return render_template("user.html")

@app.route("/report")
def report():
    return render_template("report.html")


import joblib
import numpy as np
loaded_model=joblib.load("saved_model")
@app.route('/access')
def access():
    return render_template('index.html')
    
@app.route('/',methods=['POST'])
def products():
   name=request.form.get('inputGroup-sizing-default')
   return render_template('pass.html',name=name)


@app.route('/uploader',methods=['POST'])
def getdata():
   feature=[]
   for x in request.form.values():
      feature.append(int(x)) 
   arr=[np.array(feature,dtype=np.float32)]  
   res=int(loaded_model.predict(arr)) 
   if(res==0):
     str="You are suffering from anxiety\nYou need to seek a pscychological support"
   if(res==1):
     str="You are suffering from depression\nYou need to seek a pscychological support"
   if(res==2):
     str="You are suffering from lonliness\n Speak to a counsellor or spend some time with your loved ones"
   if(res==3):
     str="You are normal"
   if(res==4):
      str="You are stressed\npractice yoga or talk to a counsellor" 
   return render_template('predict.html',label=str)

from flask import Flask, render_template, request
import tkinter as tk
import time
import speech_recognition as sr
import pickle
import joblib

with open('voice.pkl', 'rb') as file:
    model = joblib.load(file)

emotion_names = {
    'anger': 'BORDERLINE PERSONALITY DISORDER',
    'disgust': 'POST-TRAUMATIC STRESS DISORDER',
    'fear': 'NERVOUS',
    'joy': 'Normal',
    'neutral': 'SOCIAL ANXIETY DISORDER',
    'sadness': 'BIPOLAR DISORDER',
    'surprise': 'Depressed'
}

@app.route('/')
def voice():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    
    recognizer = sr.Recognizer()
    '''recording the sound '''

    with sr.Microphone() as source:
        print("Adjusting noise ")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Recording for 5 seconds")
        recorded_audio = recognizer.listen(source, timeout=5)
        print("Done recording")

    '''Recorgnizing the Audio '''
    try:
        print("Recognizing the text")
        text = recognizer.recognize_google(
                recorded_audio, 
                language="en-US"
            )
        print("Decoded Text : {}".format(text))
        prediction = model.predict([text])[0]
        print(prediction)
        prediction = model.predict([text][0])
        emotion_name = emotion_names.get(prediction, prediction)
        print(emotion_name)
    except Exception as ex:
        print(ex)
        prediction = model.predict([text])[0]
        emotion_name = emotion_names.get(prediction, prediction)
        return render_template('report.html', emotion_name=emotion_name)

import cv2
import pickle
import numpy as np
from deepface import DeepFace
import flask
from flask import Flask, render_template, request, jsonify
import base64

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
@app.route('/')
def face():
    return render_template('report.html')

@app.route('/detect_expression', methods=['POST'])
def detect_expression():
    img_data = request.get_json()['image']

    img_data = base64.b64decode(img_data.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (320, 240))

    result = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=True)

 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    faces_arr=[]
    for (x, y, w, h) in faces:
        faces_arr.append([x, y, x+w, y+h])

    with open('resultss.pkl', 'wb') as f:
        pickle.dump(result, f)

    emotion = result[0]["dominant_emotion"]
    emotion_names = {
        'angry': 'BORDERLINE PERSONALITY DISORDER',
        'disgust': 'POST-TRAUMATIC STRESS DISORDER ',
        'fear': 'NERVOUS ',
        'happy': 'Normal',
        'neutral': 'SOCIAL ANXIETY DISORDER',
        'sad': 'BIPOLAR DISORDER',
        'surprise': 'Depressed'
    }
    emotion = emotion_names.get(emotion, 'Unknown')
    return jsonify({'expression': emotion})

if __name__ == "__main__":
    app.debug='True'
    app.run()