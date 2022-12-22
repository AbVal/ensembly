import io
import numpy as np
import pandas as pd
import plotly.express as px
from flask import Flask, render_template, request, redirect, url_for
from ensembles import RandomForestMSE, GradientBoostingMSE


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
        target = request.form['target']

        p = parse_params(request.form['n_estimators'],
                         request.form['feature_scale'],
                         request.form['max_depth'],
                         request.form['learning_rate'],
                         model_name)
        n_estimators, feature_scale, max_depth, learning_rate, parse_errors = p

        if len(parse_errors) > 0:
            return render_template('index.html',
                                   n_estimators=n_estimators,
                                   feature_scale=feature_scale,
                                   max_depth=max_depth,
                                   learning_rate=learning_rate,
                                   target=target,
                                   errors=parse_errors)

        X_train, X_val, df_errors = read_dataframes(train, valid, target)

        if len(df_errors) > 0:
            return render_template('index.html',
                                   n_estimators=n_estimators,
                                   feature_scale=feature_scale,
                                   max_depth=max_depth,
                                   learning_rate=learning_rate,
                                   target=target,
                                   errors=df_errors)

        X_train, y_train = process_dataframe(X_train, target)

        X_val, y_val = process_dataframe(X_val, target)
        if X_val is not None:
            X_val = X_val.to_numpy()
            y_val = y_val.to_numpy()

        model, hist = train_model(model_name,
                                  n_estimators,
                                  feature_scale,
                                  max_depth,
                                  learning_rate,
                                  X_train.to_numpy(),
                                  y_train.to_numpy(),
                                  X_val,
                                  y_val)

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
        try:
            X_test = pd.read_csv(test)
            X_test, _ = process_dataframe(X_test)
        except Exception as err:
            return render_template('no_test.html', err=err)

        y_pred = model.predict(X_test.to_numpy())

        display_df = request.form['display_df']

        if display_df == 'yes':
            X_test['Предсказание'] = y_pred

            return render_template('results.html',
                                   tables=[X_test.to_html(classes='Загруженная выборка')],
                                   titles=X_test.columns.values)

        pred_df = pd.DataFrame(y_pred, columns=['prediction']).round(0)
        return render_template('results.html',
                               tables=[pred_df.to_html(classes='Загруженная выборка')],
                               titles=pred_df.columns.values)

    return redirect(url_for('start_page'))


def parse_params(n_estimators, feature_scale, max_depth, learning_rate, model_name):
    """
    n_estimators : string
    feature_scale : string
    max_depth : string
    learning_rate : string
    model_name : string

    Returns
    -------
    n_estimators : int
    feature_scale : float
    max_depth : int or None
    learning_rate : float
    errors : list
    """
    errors = []
    try:
        n_estimators = int(n_estimators)
        if n_estimators <= 0:
            raise ValueError
    except ValueError:
        errors.append('Некорректное количество деревьев')

    try:
        feature_scale = float(feature_scale)
        if feature_scale <= 0 or feature_scale > 1:
            raise ValueError
    except ValueError:
        errors.append('Некорректная доля признаков')

    try:
        if max_depth == '':
            max_depth = None
        else:
            max_depth = int(max_depth)
            if max_depth <= 0:
                raise ValueError
    except ValueError:
        errors.append('Некорректная глубина')

    try:
        if learning_rate == '':
            if model_name == 'GB':
                errors.append('Отсутствует темп обучения (выбран градиентный бустинг)')
            learning_rate = 1
        else:
            learning_rate = float(learning_rate)
            if learning_rate <= 0:
                raise ValueError
    except ValueError:
        errors.append('Некорректный темп обучения')

    return n_estimators, feature_scale, max_depth, learning_rate, errors


def read_dataframes(train_path, valid_path, target):
    """
    train_path : string
    valid_path : string
    target : string

    Returns
    -------
    X_train : pandas DataFrame or None
    X_val : pandas DataFrame or None
    errors : list
    """
    errors = []
    X_train, X_val = None, None
    try:
        X_train = pd.read_csv(train_path)
    except pd.errors.EmptyDataError:
        errors.append('Отсутствует обучающая выборка')
    except Exception as err:
        errors.append(f'Ошибка {err}, {type(err)}')

    try:
        X_val = pd.read_csv(valid_path)
    except pd.errors.EmptyDataError:
        pass
    except Exception as err:
        errors.append(f'Ошибка {err}, {type(err)}')

    if X_train is not None and target not in X_train.columns:
        errors.append('Таргета нет в обучающей выборке')

    if X_val is not None:
        if target not in X_val.columns:
            errors.append('Таргета нет в валидационной выборке')

        if X_train is not None:
            for col in X_train.columns:
                if col not in X_val.columns:
                    errors.append('Различные признаки обучающей и валидационной выборки')

    return X_train, X_val, errors


def process_dataframe(df, target=None):
    """
    df : pandas DataFrame
    target : string

    Returns
    -------
    X : pandas DataFrame or None
    y : pandas DataFrame or None
    """
    if df is None:
        return None, None

    y = None

    X = df
    if target is not None:
        y = df[target]
        X = df.drop(target, axis=1)

    if 'date' in df.columns:
        date = pd.to_datetime(X['date'])
        X['day'] = date.dt.day
        X['month'] = date.dt.month
        X['year'] = date.dt.year
        X = X.drop('date', axis=1)

    X = X.drop(X.columns.values[X.dtypes == 'object'], axis=1)
    return X, y


def train_model(model_name, n_estimators, feature_scale, max_depth, learning_rate, X_train, y_train, X_val, y_val):
    """
    model_name : string
    n_estimators : int
    feature_scale : float
    max_depth : int
    learning_rate : float
    X_train : numpy ndarray
    y_train : numpy ndarray
    X_val : numpy ndarray
    y_val : numpy ndarray

    Returns
    -------
    X : pandas DataFrame or None
    y : pandas DataFrame or None
    """
    if model_name == 'RF':
        model = RandomForestMSE(n_estimators=n_estimators,
                                max_depth=max_depth,
                                feature_subsample_size=feature_scale)

    # learning_rate exception
    if model_name == 'GB':
        model = GradientBoostingMSE(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    feature_subsample_size=feature_scale,
                                    learning_rate=learning_rate)

    hist = model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    return model, hist


def make_picture(history):
    """
    history : dict

    Returns
    -------
    fig : plotly express figure
    """
    n_estimators = np.arange(len(history['rmse_train']))

    df_dict = {'Число деревьев': n_estimators,
               'Обучающая выборка': history['rmse_train']}

    y_labels = ['Обучающая выборка']

    if len(history['rmse_val']) > 0:
        df_dict['Валидационная выборка'] = history['rmse_val']
        y_labels.append('Валидационная выборка')

    df = pd.DataFrame(df_dict)

    fig = px.line(df, x='Число деревьев',
                  y=y_labels,
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
