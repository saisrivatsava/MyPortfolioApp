from flask import Flask, render_template,request
from models import baseClassifier
import pandas as pd
import os
from werkzeug.utils import secure_filename
from flask import send_from_directory


UPLOAD_FOLDER = 'inputFiles'
DOWNLOAD_FOLDER = 'outputFiles'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = 'outputFiles'


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


@app.route('/nimbus', methods=['GET', 'POST'])  # , methods=['GET', 'POST'])
def show_nimbus():
    return render_template('nimbus.html')


@app.route('/load_nimbus', methods=['POST'])
def load_nimbus():
    # form =request.form
    request.form.get('name')
    pname = request.form.get('project_name')
    request.form.get('email')
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    file_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    target_feature = request.form['target_feature']
    model_type = request.form['inputState']
    if model_type == "Classification":
        acc_metrics = "toDo"
    else:
        acc_metrics = "R Squared"
    is_having_index_col = request.form.get("isHavingIndexCol")
    have_features_to_exclude = request.form.get("haveFeaturesToExclude")

    features_to_exclude_list = "none"
    if have_features_to_exclude == "yes":
        features_to_exclude_list = request.form.get("features_to_exclude_list")

    if is_having_index_col == "yes":
        data = pd.read_csv(file_url, encoding='latin-1', index_col=0)
    else:
        data = pd.read_csv(file_url, encoding='latin-1')
    test_file_flag = request.form.get("outputDataFlag")

    if test_file_flag != "yes":
        test_file_flag = "no"

    classifier = baseClassifier.BaseClassifer()

    if test_file_flag == "no" and target_feature in data.columns:
        runDetails, sorted_scores_map = classifier.trainWithoutTestFile(
            data, target_feature, features_to_exclude_list, model_type)
        os.remove(file_url)
        del classifier
        return render_template(
            'nimbus_output.html',
            sorted_scores_map=sorted_scores_map,
            runDetails=runDetails,
            pname=pname, acc_metrics=acc_metrics)

    elif test_file_flag == "yes" and target_feature in data.columns:
        test_file = request.files['test_file']
        test_filename = secure_filename(test_file.filename)
        test_file.save(
            os.path.join(
                app.config['UPLOAD_FOLDER'],
                test_filename))
        test_file_url = os.path.join(
            app.config['UPLOAD_FOLDER'], test_filename)
        target_features_to_include = request.form.get(
            'target_features_to_include')
        test_file_df = pd.read_csv(test_file_url, encoding='latin-1')

        runDetails, sorted_scores_map, models_args_list, generatedFileName, cross_val_score_mean, highestScoredModelName = classifier.trainWithTestFile(
            data, test_file_df, target_feature, target_features_to_include, pname, features_to_exclude_list, model_type)

        os.remove(file_url)
        os.remove(test_file_url)
        del classifier
        return render_template(
            'nimbus_output2.html',
            runDetails=runDetails,
            sorted_scores_map=sorted_scores_map,
            models_args_list=models_args_list,
            pname=pname,
            cross_val_score_mean=cross_val_score_mean,
            highestScoredModelName=highestScoredModelName,
            generatedFileName=generatedFileName, acc_metrics=acc_metrics)

    else:
        os.remove(file_url)
        del classifier
        return render_template(
            'nimbus_error.html',
            target_feature=target_feature,
            pname=pname)


@app.route(
    '/downloadPredicted/<path:filename>',
    methods=[
        'GET',
        'POST'])  # this is a job for GET, not POST
def download_csv(filename):
    response = send_from_directory(
        app.config['DOWNLOAD_FOLDER'],
        filename,
        as_attachment=True,
        cache_timeout=0)
    os.remove(os.path.join('outputFiles', filename))
    return response


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(debug=True)
    
    
