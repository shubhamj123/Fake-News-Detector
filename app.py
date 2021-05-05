from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
app = Flask(__name__)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = tf.keras.models.load_model('fake_news_detection.h5')
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Headline = str(request.form['Headline'])
        Content=str(request.form['Content'])
        text = Headline + " " + Content
        sequence = tokenizer.texts_to_sequences([text.lower()])
        padded_sequences = pad_sequences(sequence,padding = 'post',truncating = 'post', maxlen = 512)
        prediction=model.predict(padded_sequences)
        output = round(prediction[0][0],2)
        if output > 0.5:
            out_str = "Beware. It's a fake news!"
            out_img = 'static/fake.jpg'
            #return render_template('index.html',prediction_texts="Be Alert. It's a fake news!")
        else:
            out_str = "It's a real news!"
            out_img = 'static/fact.jpg'
            #return render_template('index.html',prediction_text="It's a real news!")
        return render_template('index.html',prediction_text="{}".format(out_str),img_path = "{}".format(out_img))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

