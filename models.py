import pandas as pd
import random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, log_loss
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
import xgboost as xgb
import optuna

import utils

class Models():

    def __init__(self, n_estimators=100, max_iter=1000, device='cpu', random_state=utils.RANDOM_SEED):
        self.n_estimators = n_estimators # Number of XGB estimators
        self.device = device # Device to use for tensorflow
        self.max_iter = max_iter # Max iterations for LR and SGD
        self.random_state = random_state # Random state for XGB
        self.models = self.define_models()
        self.len_models = len(self.models)

    def init_models(self):

        # XGB paramters, booster type and hyperparameter values obtained from optuna
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

        models = {'xgb' : xgb.XGBClassifier(**xgb_params),
                  'lr' : LogisticRegression(max_iter=self.max_iter),
                  'sgd' : SGDClassifier(loss='log_loss', max_iter=self.max_iter),
                  'ann' : ann
                 }
        
        return models