import io
import numpy as np
import pandas as pd
import plotly.express as px
from flask import Flask, render_template, request, redirect, url_for
from ensembles import RandomForestMSE, GradientBoostingMSE

# TODO:
# docstrings
# css

app = Flask(__name__)
model = None


@app.route('/', methods=['GET', 'POST'])
def start_page():
    return render_template('index.html')


@app.route('/train', methods=['GET', 'POST'])
def train():
    global model
    if request.method == 'POST':
        train = request.files['train']
        valid = request.files['valid']

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

        X_train = pd.read_csv(train)
        X_valid = pd.read_csv(valid)

        model, hist = train_model(model_name, n_estimators, feat, max_depth, learning_rate, X_train, X_valid)

        buffer = io.StringIO()
        fig = make_picture(hist)
        fig.write_html(buffer)
        graph_html = buffer.getvalue()

        return render_template('predict.html',
                               n_estimators=model.n_estimators,
                               feature_scale=model.feature_subsample_size,
                               max_depth=model.max_depth,
                               learning_rate=learning_rate,
                               model_name=model_name,
                               tables=[X_train.sample(100).to_html(classes='Обучающая выборка')],
                               titles=X_train.columns.values,
                               graph=graph_html)

    return redirect(url_for('start_page'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global model
    if request.method == 'POST':
        test = request.files['test']
        X_test = pd.read_csv(test)
        X_test_numpy = X_test.drop(X_test.columns.values[X_test.dtypes == 'object'], axis=1).to_numpy()

        y_pred = model.predict(X_test_numpy)

        display_df = request.form['display_df']

        if display_df == 'yes':
            X_test['Предсказание'] = y_pred

            return render_template('results.html',
                                   tables=[X_test.to_html(classes='Загруженная выборка')],
                                   titles=X_test.columns.values)

        pred_df = pd.DataFrame(y_pred, columns=['prediction'])
        return render_template('results.html',
                               tables=[pred_df.to_html(classes='Загруженная выборка')],
                               titles=pred_df.columns.values)

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


def make_picture(history):
    n_estimators = np.arange(len(history['rmse_train']))

    df_dict = {'Число деревьев': n_estimators,
               'Валидационная выборка': history['rmse_val'],
               'Обучающая выборка': history['rmse_train']}

    df = pd.DataFrame(df_dict)

    fig = px.line(df, x='Число деревьев',
                  y=['Обучающая выборка', 'Валидационная выборка'],
                  title="RMSE от количества деревьев",
                  labels={'x': 'Число деревьев', 'value': 'RMSE'}, height=600)

    fig.update_layout(legend=dict(
        title='',
        yanchor='top',
        y=1,
        xanchor='right',
        x=1
    ))

    return fig


if __name__ == '__main__':
    app.run(debug=True)
