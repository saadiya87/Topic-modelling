from flask import Flask,render_template,request,url_for
import pickle
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/",methods=['POST'])
def predict():
    url = "final1.csv"
    df = pd.read_csv(url)
    df_data = df[["clean_text","Dominant_Topic"]]
    X = df_data['clean_text'].astype('U')
    y = df_data.Dominant_Topic
    #X_train, X_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2,shuffle=True)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    model  = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X,y)
    labels = model.predict(X)
    
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        #vect = data.astype('U')
        my_prediction = model.predict(data)	
    return render_template('results.html',prediction = my_prediction,comment = comment)

        
if __name__ == '__main__':
	app.run(debug=True)
    

        

        

    
    
    
    
    