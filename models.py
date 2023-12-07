import pandas as pd
import random
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import xgboost as xgb
import optuna

import utils



class Models():

    def __init__(self, n_estimators=100, max_iter=1000, random_state=utils.RANDOM_SEED):
        self.n_estimators = n_estimators # Number of XGB estimators
        self.max_iter = max_iter # Max iterations for LR and SGD
        self.random_state = random_state # Random state for XGB
        self.models = self._init_models()
        self.n_models = len(self.models)



    def _init_models(self):

        # XGB paramters, hyperparameter values obtained from optuna
        xgb_params = {'objective': 'binary:logistic',
                      'tree_method': 'hist',
                      'eval_metric': 'logloss',
                      'booster': "gblinear",
                      'lambda': 3.76652651097892e-05,
                      'alpha': 2.270724881190875e-06,
                      'subsample': 0.919826926255599,
                      'colsample_bytree': 0.9634746156545292,
                      'verbosity': 0
                     }
        
        # ANN with 4 x 800 fully connected layers
        ann = tf.keras.models.Sequential([
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(800, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(800, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(800, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(800, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(1, activation='sigmoid')
                ])
        
        # Set ANN optimizer and loss function
        optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)
        loss_func = tf.keras.losses.BinaryCrossentropy()

        # Compile ANN
        ann.compile(optimizer=optimizer, loss = loss_func, metrics=['binary_accuracy'])

        # Return dictionary of all models
        models = {'xgb' : xgb.XGBClassifier(**xgb_params),
                  'lr' : LogisticRegression(max_iter=self.max_iter),
                  'sgd' : SGDClassifier(loss='log_loss', max_iter=self.max_iter),
                  'ann' : ann
                 }
        
        return models
    

    
    def train(self, train_set, train_labels):

        # Prepare XGB input
        dm_train = xgb.DMatrix(train_set, train_labels)

        # Prepare ANN input
        tf_data = tf.convert_to_tensor(train_set, dtype='float32')
        tf_labels = tf.convert_to_tensor(train_labels, dtype='float32')

        # Prepare ANN save callback
        tf_save_callback = tf.keras.callbacks.ModelCheckpoint(filepath='output/ann.ckpt',
                                                              save_weights_only=True,
                                                              verbose=0)

        # Train models
        self.models['xgb'].fit(train_set, train_labels)
        self.models['lr'].fit(train_set, train_labels)
        self.models['sgd'].fit(train_set, train_labels)
        self.models['ann'].fit(x=tf_data, 
                               y=tf_labels, 
                               epochs=50, 
                               batch_size=64,
                               callbacks=[tf_save_callback])

        # Save models
        self.models['xgb'].save_model("output/xgb.json")
        joblib.dump(self.models['lr'], 'output/lr.joblib')
        joblib.dump(self.models['sgd'], 'output/sgd.joblib')



    def evaluate(self, test_set, test_labels):

        # Load models
        xgb_bst = xgb.Booster(model_file='output/xgb.json')
        self.models['lr'] = joblib.load('output/lr.joblib')
        self.models['sgd'] = joblib.load('output/sgd.joblib')
        self.models['ann'].load_weights('output/ann.ckpt')

        # Prepare ANN input
        tf_data = tf.convert_to_tensor(test_set, dtype='float32')
        tf_labels = tf.convert_to_tensor(test_labels, dtype='float32')


        # Get XGB, LR and SGD predictions
        xgb_pred = xgb_bst.predict(test_set)
        lr_pred = self.models['lr'].predict(test_set)
        sgd_pred = self.models['sgd'].predict(test_set)
        
        # Compute binary accuracy
        xgb_acc = accuracy_score(test_labels, xgb_pred)
        lr_acc = accuracy_score(test_labels, lr_pred)
        sgd_acc = accuracy_score(test_labels, sgd_pred) 
        self.models['ann'].evaluate(tf_data, tf_labels)

        print("XGB accuracy: ", xgb_acc)
        print("Logistic regression accuracy: ", lr_acc)
        print("SGD accuracy: ", sgd_acc)
        



    def predict(self, input_df):

        # Assume input is a pd dataframe
        n_preds = len(input_df.index)
        # Transpose dataframe in case of single input to get correct shape
        if n_preds == 1:
            input_df = input_df.T

        # Load models
        xgb_bst = xgb.Booster(model_file='output/xgb.json')
        self.models['lr'] = joblib.load('output/lr.joblib')
        self.models['sgd'] = joblib.load('output/sgd.joblib')
        self.models['ann'].load_weights('output/ann.ckpt')

        # Prepare ANN input
        tf_input = tf.convert_to_tensor(input_df, dtype='float32')

        # Get predictions
        xgb_pred = xgb_bst.predict_proba(input_df)
        lr_pred = self.models['lr'].predict_proba(input_df)
        sgd_pred = self.models['sgd'].predict_proba(input_df)
        ann_pred = self.models['ann'].predict(tf_input)


        predictions = {'XGB' : xgb_pred,
                       'LR' : lr_pred,
                       'SGD' : sgd_pred,
                       'ANN' : ann_pred
                      }
        
        for i in range(n_preds):
            aggregate_pred = (ann_pred[i][0] 
                              + lr_pred[i][1]
                              + sgd_pred[i][1]
                              + xgb_pred[i][0]) / 4
            print('Averaged win probability prediction: {:.3f}'.format(aggregate_pred))

        return predictions