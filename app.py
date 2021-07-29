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
	# Link to dataset
	#url = "final.csv"
	#df = pd.read_csv(url)
	#df_data = df[["Keywords","Dominant_Topic"]]
	# Features and Labels
	#df_x = df_data['Keywords']
	#df_y = df_data.Dominant_Topic
    # Extract Feature With CountVectorizer
	#corpus = df_x
	encoder = LabelEncoder()
	#X = cv.fit_transform(corpus) # Fit the Data
	#from sklearn.model_selection import train_test_split
	#X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.25, random_state=42)
	#Naive Bayes Classifier
	#from sklearn.neighbors import KNeighborsClassifier
	#clf = KNeighborsClassifier()
	#clf.fit(X_train.values.reshape(-1, 1),y_train)
	#clf.score(X_test,y_test)
    
	#Alternative Usage of Saved Model
	ytb_model = open("model.pkl","rb")
	clf = joblib.load(ytb_model)

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = encoder.fit_transform(data)
		vect = vect.reshape(1, -1)
		my_prediction = clf.predict(vect)
	return render_template('results.html',prediction = my_prediction,comment = comment)
	


if __name__ == '__main__':
	app.run(debug=True)