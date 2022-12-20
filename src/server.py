import uuid
import numpy as np
import pandas as pd
import plotly.express as px
from flask import Flask, render_template, request, redirect, url_for
from ensembles import RandomForestMSE, GradientBoostingMSE

# TODO:
# predictions
# docstrings
# css

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def start_page():
    return render_template('index.html')


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        train = request.files['train']
        valid = request.files['valid']
        showflag = request.form.get('show_graph')
        if showflag is None:
            showflag = False

        model_name = request.form['model']

        try:
            n_estimators = int(request.form['n_estimators'])

            feat = float(request.form['feature_scale'])

            max_depth = request.form['max_depth']
            if max_depth == '':
                max_depth = None
            else:
                max_depth = int(request.form['max_depth'])

            learning_rate = request.form['learning_rate']
            if learning_rate == '':
                if model_name == 'GB':
                    return redirect(url_for('start_page'))
                learning_rate = 1
            else:
                learning_rate = float(learning_rate)
        except:
            return redirect(url_for('start_page'))

        if n_estimators <= 0:
            return redirect(url_for('start_page'))
        if feat <= 0 or feat > 1:
            return redirect(url_for('start_page'))
        if max_depth is not None and max_depth <= 0:
            return redirect(url_for('start_page'))

        # df = pd.read_csv(train)
        # TODO:
        # dataframe view
        # return render_template('pd_df.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)

        X_train = pd.read_csv(train)
        X_valid = pd.read_csv(valid)

        model, hist = train_model(model_name, n_estimators, feat, max_depth, learning_rate, X_train, X_valid)

        random_string = uuid.uuid4().hex
        path = "static/temp/" + random_string + ".svg"

        make_picture(hist, path, showflag)

        return render_template('predict.html', href=path,
                               n_estimators=model.n_estimators,
                               feature_scale=model.feature_subsample_size,
                               max_depth=model.max_depth,
                               learning_rate=learning_rate,
                               model_name=model_name,
                               tables=[X_train.sample(100).to_html(classes='Обучающая выборка')],
                               titles=X_train.columns.values)

    return redirect(url_for('start_page'))


# TODO:
# send arguments through url
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        test = request.files['test']
    return redirect(url_for('start_page'))


def train_model(model_name, n_estimators, feat, max_depth, learning_rate, X_train, X_valid):
    y_train = X_train.price.to_numpy()
    X_train = X_train.drop('price', axis=1)
    X_train = X_train.drop(X_train.columns.values[X_train.dtypes == 'object'], axis=1).to_numpy()

    if X_valid is not None:
        y_valid = X_valid.price.to_numpy()
        X_valid = X_valid.drop('price', axis=1)
        X_valid = X_valid.drop(X_valid.columns.values[X_valid.dtypes == 'object'], axis=1).to_numpy()

    if model_name == 'RF':
        model = RandomForestMSE(n_estimators=n_estimators,
                                max_depth=max_depth,
                                feature_subsample_size=feat)

    # learning_rate exception
    if model_name == 'GB':
        model = GradientBoostingMSE(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    feature_subsample_size=feat,
                                    learning_rate=learning_rate)

    hist = model.fit(X_train, y_train, X_val=X_valid, y_val=y_valid)
    return model, hist


def make_picture(history, output_file, showflag):
    n_estimators = np.arange(len(history['rmse_train']))

    df_dict = {'Число деревьев': n_estimators,
               'Валидационная выборка': history['rmse_val'],
               'Обучающая выборка': history['rmse_train']}

    df = pd.DataFrame(df_dict)

    fig = px.line(df, x='Число деревьев',
                  y=['Обучающая выборка', 'Валидационная выборка'],
                  title="RMSE от количества деревьев",
                  labels={'x': 'Число деревьев', 'value': 'RMSE'})

    fig.update_layout(legend=dict(
        title='',
        yanchor='top',
        y=1,
        xanchor='right',
        x=1
    ))

    fig.write_image(output_file, width=800, engine='kaleido')

    if showflag:
        fig.show()

if __name__ == '__main__':
    app.run(debug=True)
