
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

# verbose = 1
# shuffle = True
metric = ['accuracy']
# metric = [tf.keras.metrics.Recall()]

# folder setting
# model_folder = 'model'
# training_folder = 'training_history'
# model_suffix = '_best_model'

def callback(model_folder = 'model',
             training_folder = 'training_history',
             monitor_metric = 'accuracy',
             model_name = None):
    callbacks_list = [
            ModelCheckpoint(
                filepath = os.path.join(model_folder, model_name+'.h5'),
                monitor='val_accuracy',
                verbose = 1,
                save_best_only=True
            ),
            # EarlyStopping(
            #     monitor = monitor_metric,
            #     # patience = np.floor(epochs)
            # ),
            CSVLogger(
                filename = os.path.join(training_folder, model_name+'.csv'),
                separator=',',
                append=False
            )
        ]
    return callbacks_list

# train model function
def model_train(X_train, y_train, model ,
#                 X_valid = X_valid, y_valid = y_valid,
                epochs = 10, shuffle = True, batch_size = 64,
                prefix = '', model_suffix = '_best_model',
                class_weight=None, loss = 'categorical_crossentropy', opt ='adam', metric = ['accuracy']
                ):

    model.compile(
        loss = loss,
        optimizer = opt,
        metrics = metric
    )

    model.fit(
        X_train, y_train,
        epochs = epochs,
        batch_size = batch_size,
        verbose = 1,
        callbacks = callback(model_name = prefix + model_suffix, monitor_metric = 'accuracy'),
        shuffle = shuffle,
        class_weight = None,
        validation_split = 0.33 ,
#         validation_data = (X_valid,y_valid)
    )
    return
