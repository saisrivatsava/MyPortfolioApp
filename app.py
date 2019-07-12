from flask import Flask, render_template
from flask import Flask, render_template, flash, request, url_for
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from models import classification_models
import pandas as pd
import os
from werkzeug.utils import secure_filename


UPLOAD_FOLDER = 'inputFiles'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/portfolio')
def portfolio():
   return render_template('portfolio.html')


@app.route('/nimbus', methods=['GET','POST'])#, methods=['GET', 'POST'])
def show_nimbus():
    return render_template('nimbus.html')

@app.route('/load_nimbus', methods=['POST'])
def load_nimbus():
    # form =request.form
    name=request.form.get('name')
    pname=request.form.get('project_name')
    email=request.form.get('email')
    file=request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    file_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)#url_for('uploaded_file',filename=filename)
    target_feature=request.form['target_feature']
    inputState=request.form['inputState']
    data = pd.read_csv(file_url,encoding='latin-1')
    if target_feature in data.columns:
        classifier = classification_models.Classifier()
        runDetails, result = classifier.train(data, target_feature)
        os.remove(file_url)
        return render_template('nimbus_output.html',result=result, runDetails=runDetails, pname=pname)
    else:
        os.remove(file_url)
        return render_template('nimbus_error.html',target_feature=target_feature, pname=pname)






if __name__ == "__main__":
    app.run(debug=True)
