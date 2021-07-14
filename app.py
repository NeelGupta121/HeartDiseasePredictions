import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle


app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # for rendering result from html gui
    features=[x if float(x)==int(float(x)) else float(x) for x in request.form.values()]

    final_features= np.array(features)
    
    final_features=final_features.astype(np.float64)
    
    final=final_features.reshape(1,-1)
    
    predictions=model.predict(final)
    
    if(predictions[0]==1):
        return render_template('index.html', prediction_text='The person has a Heart Disease')
    else:
        return render_template('index.html', prediction_text='The person has a Healthy Heart')



if(__name__=="__main__"):
    app.run(debug=True)