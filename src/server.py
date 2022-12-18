from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from ensembles import RandomForestMSE, GradientBoostingMSE
import plotly.express as px
import plotly.graph_objects as go
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/', methods=['GET'])
def start_page():
    return render_template('index.html')


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
        train = request.files['train']
        valid = request.files['valid']
        print(train)
        print(valid)
        n_estimators = request.form['n_estimators']
        feat = request.form['feature_scale']
        max_depth = request.form['max_depth']
        learning_rate = request.form['learning_rate']

        print(n_estimators, feat, max_depth, learning_rate)
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)