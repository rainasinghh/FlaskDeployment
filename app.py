from flask import Flask, request,render_template
import numpy as np
import pickle

app = Flask(__name__)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # Change int to float
    int_features = [float(x) for x in request.form.values()]  
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = prediction[0]  # Get the single prediction value
    return render_template('index.html', 
        prediction_text='Daughter\'s height should be {:.2f}'.format(output))


if __name__ == "__main__": 
    app.run(port=5000)