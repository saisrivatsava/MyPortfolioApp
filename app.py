from flask import Flask, render_template
from flask import Flask, render_template, flash, request, url_for
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from models import classification_models
import pandas as pd
import os
from werkzeug.utils import secure_filename
from flask import send_file


UPLOAD_FOLDER = 'inputFiles'

outputFileName= ""

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
    test_file_flag = request.form.get("outputDataFlag")

    if test_file_flag !="yes":
        test_file_flag="no"

    classifier = classification_models.Classifier()

    if test_file_flag=="no" and target_feature in data.columns:
        runDetails, result = classifier.train(data, target_feature)
        os.remove(file_url)
        return render_template('nimbus_output.html',result=result, runDetails=runDetails, pname=pname)

    elif test_file_flag=="yes" and target_feature in data.columns:
        test_file=request.files['test_file']
        test_filename = secure_filename(test_file.filename)
        test_file.save(os.path.join(app.config['UPLOAD_FOLDER'], test_filename))
        test_file_url = os.path.join(app.config['UPLOAD_FOLDER'], test_filename)
        target_features_to_include = request.form.get('target_features_to_include')
        test_file_df = pd.read_csv(test_file_url,encoding='latin-1')

        runDetails,sorted_scores_map, models_args_list, generatedFileName = classifier.trainWithTestData(data, test_file_df, target_feature,  target_features_to_include, pname)
        global outputFileName
        outputFileName = generatedFileName
        print("opfile "+outputFileName)
        os.remove(file_url)
        return render_template('nimbus_output2.html', runDetails=runDetails,sorted_scores_map=sorted_scores_map, models_args_list=models_args_list, pname=pname)

    else:
        os.remove(file_url)
        return render_template('nimbus_error.html',target_feature=target_feature, pname=pname)

    classifier.__del__(self)


@app.route('/downloadPredicted') # this is a job for GET, not POST
def download_csv():
    print("inside download meth")
    print(outputFileName)
    return send_file(outputFileName,
                     mimetype='text/csv',
                     attachment_filename=outputFileName,
                     as_attachment=True)

# os.remove(outputFileName)

if __name__ == "__main__":
    app.run(debug=True)
