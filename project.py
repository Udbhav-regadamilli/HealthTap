from flask import Flask, render_template, request
import joblib
import spacy
import pandas as pd

app = Flask(__name__, static_folder='templates/assets')
model = joblib.load(open('./DiseasePredictionBasedonSymptoms_Pipeline.joblib', 'rb'))

nlp = spacy.load('en_core_web_sm')

df = pd.read_csv('data.csv')
data = df.values.tolist()
print(data)

# Preprocessing function
def preprocess(symptoms):
    processed_symptoms = []
    for symptom in symptoms:
        doc = nlp(symptom)
        processed_symptom = ' '.join(token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha)
        processed_symptoms.append(processed_symptom)
    return ' '.join(processed_symptoms)

@app.route('/')
@app.route('/index.html')
def index():
    return render_template("index.html")

@app.route('/predict.html', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        sample_symptom = request.form['s1'].strip() if request.form['s1'].strip() != "" else '' 
        sample_symptom += ',' + request.form['s2'].strip() if request.form['s2'].strip() != "" else '' 
        sample_symptom += ',' + request.form['s3'].strip() if request.form['s3'].strip() != "" else ''
        print(sample_symptom + " ")
        print(request.form['s1'])
        print(request.form['s2'])
        print(request.form['s3'])
        processed_symptom = preprocess([sample_symptom])
        prediction = model.predict([processed_symptom])
        msgs = []
        for i in range(len(data)):
            if(data[i][0].lower() == str(prediction[0]).lower()):
                msgs =  data[i][1:]
                break
                    

        print("Predicted disease:", prediction)
        return render_template("predict.html", prediction_text='You may have been diagnosed with '+prediction[0], msg=msgs, len=len(msgs))
    return render_template("predict.html", prediction_text='', msg=[], len=0)

if __name__ == '__main__':
    app.run(debug=True)