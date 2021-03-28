import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.metrics import confusion_matrix


def plot_col(df, num_cols):
    plot_cols_list = df.columns[:-1]
    num_cols = 3
    num_rows = (len(plot_cols_list) / num_cols)
    if not num_rows.is_integer():
        num_rows = int(num_rows + 1)
    num_rows = int(num_rows)
    sns.set_style("dark")
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 120))
    counter = 0
    for i in range(num_rows):
        for j in range(num_cols):
            try:
                col = plot_cols_list[counter]
                sns.distplot(df[col], ax=axes[i, j])
            except:
                axes[i, j].set_title('')
                counter += 1


def model_plot(model, history, Predict, y_test):
    print(model)
    sns.set_style("dark")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    df_plot = pd.read_csv(os.path.join('training_history', history))
    for col in df_plot.columns:
        if 'accuracy' in col:
            sns.lineplot(x=df_plot['epoch'], y=col, data=df_plot, label=col, ax=axes[0])
            axes[0].set_ylim(0, 1)
            axes[0].set_title('Training Accuracy')
        if 'loss' in col:
            sns.lineplot(x=df_plot['epoch'], y=col, data=df_plot, label=col, ax=axes[1])
            # axes[1].set_ylim(0, 5)
            axes[1].set_title('Training Loss')
    model_path = os.path.join('model', model)
    model = tf.keras.models.load_model(model_path)
    model.evaluate(Predict, y_test, verbose=2)
    y_pred = np.argmax(model.predict(Predict), axis=1)
    y_truth = np.argmax(y_test, axis=1)
    CM = confusion_matrix(y_truth, y_pred, labels=[x for x in range(y_test.shape[1])])
    sns.heatmap(CM, annot=True, fmt='g', square=False, cmap='Blues', annot_kws={
                "size": 20, "ha": 'center', "va": 'center'}, ax=axes[2])
    axes[2].set_title('Test Confusion Matrix')
