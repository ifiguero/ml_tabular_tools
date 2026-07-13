import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix , brier_score_loss
from sklearn.utils.multiclass import type_of_target
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
from ray import tune
import ray

from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.utils import set_random_seed
from keras.metrics import AUC
from keras.layers import Dense, BatchNormalization, Dropout
from kerastuner.tuners import RandomSearch, Hyperband, GridSearch
from kerastuner.engine.trial import TrialStatus

import shap

from datetime import datetime
import enlighten
import logging
import joblib
import zipfile
import pickle
import time
import json
import os
import re

from scipy.stats import loguniform, randint

import warnings
from sklearn.exceptions import FitFailedWarning

# Add UserWarning to the tuple
warnings.simplefilter(action='ignore', category=(FutureWarning, FitFailedWarning, UserWarning))

#tf.config.experimental.enable_op_determinism()
#from sklearn.experimental import enable_halving_search_cv  # noqa
#from sklearn.model_selection import HalvingRandomSearchCV

class BinaryTuner:
    def __init__(self, dataFrame, label_class, seeds=None, dnn=False, test_size=0.2, test_prio=0.9, tuneScoring=None, debug=False, n_seeds=3):
        self.ledger = pd.DataFrame(columns=["node", "ts", "Dataset", "Model", "Params", "Seed", "Ratio", "Accuracy", "Specificity", "Recall", "F1", "ROC_AUC"])
        self.name = label_class
        self.safe_name = re.sub(r'[^\w\-]', '', str(label_class).strip().replace(' ', '_'))

        os.makedirs(self.safe_name, exist_ok=True)
        self.start = int(time.time())

        log_format = '%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        logging.basicConfig(format=log_format, datefmt=date_format)

        target_log = '{}/load-{}.log'.format(self.safe_name, self.start)
        fh = logging.FileHandler(target_log)

        self.debug = debug
        self.test_prio = test_prio
        self.tuneScoring = tuneScoring

        if not dataFrame[label_class].isin([0, 1]).all():
            raise ValueError(f"Invalid {label_class} column values, Only 0 and 1 are allowed.")

        self.dataFrame = dataFrame.copy()
        self.dnn = dnn
        self.logger = logging.getLogger("BinaryTuners")
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
            fh.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
            fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)

        self.last_ping = self.start
        self.ratio = test_size

        if not isinstance(seeds, list):
            self.seeds = [np.random.randint(1, 2**20) for _ in range(n_seeds)]
        else:
            self.seeds = seeds

        self.logger.info('{:#^60}'.format(label_class))
        self.loadCheckPoint()
        self.__metaVars()

    def __metaVars(self):
        len_models = len(self.get_model_train()) + self.dnn
        self.logger.info("Len models: {}".format(len_models))
        len_seeds = len(self.seeds)
        self.logger.info("Len seeds: {}".format(len_seeds))



        full  = self.dataFrame.drop(self.name, axis=1)
        self.nvars = full.shape[1]
        self.logger.info("Nvars: {}".format(self.nvars))
        label_data_drop = self.dataFrame.dropna()[self.name]
        label_data_full = self.dataFrame[self.name]
        if label_data_drop.shape[0] != label_data_full.shape[0]: # Missingvals in the dataset
            valsize = int(self.nvars/2)
            self.logger.info("Valsize: {}".format(valsize))
            self.noMissingDataset = self.dataFrame.dropna().copy()
            self.missingDatasets = []
            self.missingDatasets.append((self.dataFrame.copy(), 'fulldataset'))
            self.missingDatasets.append((self.dataFrame[self.dataFrame.isna().sum(axis=1) <= valsize].copy(), 'drop{}'.format(valsize)))
            self.logger.info("Len noMissingDataset: {}".format(self.noMissingDataset.shape[0]))
            for i, df in enumerate(self.missingDatasets):
                self.logger.info("Len MissingDataset {}: {}".format(i, df[0].shape[0]))
        else:
            self.noMissingDataset = self.dataFrame.copy()
            self.missingDatasets = []

        os.makedirs("{}/nomissing-original".format(self.safe_name), exist_ok=True)
        len_datasets = 1 + 2*len(self.missingDatasets)

        self.logger.info("Len datasets: {}".format(len_datasets))

        len_unbalanced = 0

        if not self.is_balanced(self.noMissingDataset[self.name]):
            len_unbalanced += 1
            os.makedirs("{}/nomissing-oversampled".format(self.safe_name), exist_ok=True)

        for dfData, dfname in self.missingDatasets:
            os.makedirs("{}/{}-original".format(self.safe_name, dfname), exist_ok=True)
            if not self.is_balanced(dfData[self.name]):
                len_unbalanced += 2
                os.makedirs("{}/{}-oversampled".format(self.safe_name, dfname), exist_ok=True)

        self.logger.info("Len unbalanced: {}".format(len_unbalanced))

        total_models = len_seeds * len_models * (len_datasets + len_unbalanced)

        self.logger.info("Total Models to be trained: {}".format(total_models))
        self.logger.info("Total Models in the ledger: {}".format(self.trained))
        self.total_models = total_models
        self.logger.info("{:=^60}".format("######"))

    def addSeed(self, n_seeds=None, seeds=None):

        if isinstance(seeds, list):
            self.seeds = list(set(self.seeds + seeds))
        elif isinstance(n_seeds, int):
            seeds = [np.random.randint(1, 2**20) for _ in range(n_seeds)]
            self.seeds = list(set(self.seeds + seeds))
        else:
            seeds = [np.random.randint(1, 2**20)]
            self.seeds = list(set(self.seeds + seeds))

        self.saveCheckPoint()
        self.__metaVars()

    def is_balanced(self, dfData):
        value_a, value_b = dfData.value_counts()
        total_len = value_a + value_b #dataset length
        balance_ratio = int(100*abs((value_a - value_b)/(value_a + value_b)))
        return balance_ratio < 5

    def ping(self, msg):
      curtime = int(time.time())
      delta = curtime - self.last_ping
      self.last_ping = curtime
      self.logger.info("{:<50}\t|{:4}m {:2}s".format(msg, int(delta//60), int(delta%60)))

    def loadCheckPoint(self):
        if not os.path.isfile('{}/Simulaciones.xlsx'.format(self.safe_name)):
            self.saveCheckPoint()

        with pd.ExcelFile('{}/Simulaciones.xlsx'.format(self.safe_name)) as xls:
            self.ledger = pd.read_excel(xls, sheet_name='Historial')
            self.trained = self.ledger.shape[0]

        with pd.ExcelFile('{}/Dataset.xlsx'.format(self.safe_name)) as xls:
            self.dataFrame = pd.read_excel(xls, sheet_name=self.name)

        with open('{}/vars.pickle'.format(self.safe_name), 'rb') as pfile:
            self.name, self.seeds, self.dnn, self.ratio, self.test_prio, self.tuneScoring  = pickle.load(pfile)
            self.safe_name = re.sub(r'[^\w\-]', '', str(self.name).strip().replace(' ', '_'))

    def saveCheckPoint(self):
        with pd.ExcelWriter('{}/Simulaciones.xlsx'.format(self.safe_name), engine='xlsxwriter') as xls:
            self.ledger.to_excel(xls, sheet_name='Historial', index=False)

        with pd.ExcelWriter('{}/Dataset.xlsx'.format(self.safe_name), engine='xlsxwriter') as xls:
            self.dataFrame.to_excel(xls, sheet_name=self.name, index=False)

        with open('{}/vars.pickle'.format(self.safe_name), 'wb') as pfile:
            pickle.dump((self.name, self.seeds, self.dnn, self.ratio, self.test_prio, self.tuneScoring), pfile, protocol=pickle.HIGHEST_PROTOCOL)


        self.trained = self.ledger.shape[0]

    def get_model_train_keras(self, hp):

        model = Sequential()
        model.add(Dense(units=hp.Int('units_input', min_value=48, max_value=56, step=8), input_dim=self.nvars, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(rate=hp.Float('dropout_input', min_value=0.1, max_value=0.1, step=0.1)))

        model.add(Dense(units=hp.Int('units_hidden', min_value=32, max_value=48, step=8), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(rate=hp.Float('dropout_hidden', min_value=0.4, max_value=0.4, step=0.1)))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC()])

        return model

    def train_and_score_model_keras(self, X_train, X_test, y_train, y_test, seed, label):
#        set_random_seed(seed)
        ntrials = 6
        tuner = RandomSearch(
            self.get_model_train_keras,
            objective='val_loss', #val_loss
#            seed=seed,
            max_trials=ntrials,
            project_name='{}-{}'.format(label,seed))

        self.logger.info("{:~^60}".format(' {}-{} '.format(label,seed)))


        search_dir = "{}/keras-tuner-{}/".format(self.safe_name,label)
        os.makedirs(search_dir, exist_ok=True)
        search_callback = TensorBoard(log_dir=search_dir)
        early_stopping_search = EarlyStopping(monitor='val_loss', patience=13, min_delta=0.005, start_from_epoch=7, restore_best_weights=True)
        tuner.search(X_train, y_train, epochs=150, batch_size=10, validation_data=(X_test, y_test), callbacks=[early_stopping_search, search_callback])
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # best_worse = float(0)
        # model_seq = 0
        # best_hps = ''
        # optimized_model = None
        #
        # for current, trial in enumerate(tuner.oracle.get_best_trials(num_trials=ntrials)):
        #     if trial.status == TrialStatus.COMPLETED:
        #         # Retrieve the training and validation metrics for the last step
        #         auc = trial.metrics.get_last_value("auc")
        #         val_auc = trial.metrics.get_last_value("val_auc")
        #         if auc is not None and val_auc is not None:
        #             worse = min(auc, val_auc)
        #
        #             # Update the best trial if this difference is the smallest
        #             if worse > best_worse:
        #                 best_worse = worse
        #                 model_seq = current
        #                 best_auc, best_val_auc = auc, val_auc
        #                 optimized_model = tuner.load_model(trial)
        #                 best_hps = trial.hyperparameters
        #
        # self.logger.info(f"Selected trial with (auc, val_auc) : ({best_auc}, {best_val_auc})")
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        optimized_model = tuner.get_best_models(num_models=1)[0]
        # if optimized_model is None:
        #     raise('model load failed')


        # # Train the model
        # optimized_model = Sequential()
        # optimized_model.add(Dense(units=best_hps.get('units_input'), input_dim=X_train.shape[1], activation='relu'))
        # optimized_model.add(BatchNormalization())
        # optimized_model.add(Dropout(rate=best_hps.get('dropout_input')))
        # optimized_model.add(Dense(units=best_hps.get('units_hidden'), activation='relu'))
        # optimized_model.add(BatchNormalization())
        # optimized_model.add(Dropout(rate=best_hps.get('dropout_hidden')))
        # optimized_model.add(Dense(1, activation='sigmoid'))
        # optimized_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'auc'])
        #
        #
        fit_dir = "{}/keras-fit-{}/".format(self.safe_name, label)
        os.makedirs(fit_dir, exist_ok=True)
        train_callback = TensorBoard(log_dir=fit_dir)
        #
        model_params = "UI:{}, DI:{}, UH: {}, DH: {}".format(best_hps.get('units_input'), best_hps.get('dropout_input'), best_hps.get('units_hidden'), best_hps.get('dropout_hidden'))
        self.logger.info("Model Params: {}".format(model_params))
        early_stopping_train = EarlyStopping(monitor='val_loss', start_from_epoch=7, patience=43, restore_best_weights=True)
        optimized_model.fit(X_train, y_train, epochs=200, batch_size=10, validation_data=(X_test, y_test), callbacks=[early_stopping_train, train_callback])

        y_pred = optimized_model.predict(X_test)
        if type_of_target(y_pred) == "continuous":
            # make a numpy array from y_pred where all the values > 0.5 become 1 and all remaining values are 0
            y_pred = np.where(y_pred > 0.5, 1, 0)

        brier = None
        if hasattr(optimized_model, "predict_proba"):
            try:
                y_prob = optimized_model.predict_proba(X_test)[:, 1]
                brier = brier_score_loss(y_test, y_prob)
            except Exception as e:
                pass

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        npv = tn / (tn + fn)
        ppv = tp / (tp + fp)

        self.trained += 1
        self.bar.update()
        return roc_auc, f1, accuracy, recall, sensitivity, specificity, npv, ppv, brier, optimized_model, model_params

    def get_model_train(self):
        return [
            LogisticRegression(),
            XGBClassifier(),
            RandomForestClassifier(),
            Perceptron(),
            SGDClassifier(),
            SVC(),
            GaussianNB(),
            KNeighborsClassifier(),
#             GradientBoostingClassifier(),
            PLSRegression(),
            LinearDiscriminantAnalysis()
        ]

    def get_tunable_params(self, model):
        if isinstance(model, LogisticRegression):
            return {
                "C": np.logspace(-2, 2, 15),
                "max_iter": [80, 100, 150]
            }
        elif isinstance(model, XGBClassifier):
            return {
                "n_estimators": [50, 100, 200],
                "learning_rate": np.logspace(-4, -1, 8),
                "max_depth": [3, 5, 7]
            }
        elif isinstance(model, RandomForestClassifier):
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15],
                "max_features": [2, 5, 10] #['n', 'max_depth', 'max_features', 'max_leaf_nodes', 'max_samples', 'min_impurity_decrease', 'min_samples_leaf', 'min_samples_split', 'min_weight_fraction_leaf', 'monotonic_cst', 'n_estimators', 'n_jobs', 'oob_score', 'random_state', 'verbose', 'warm_start']
            }
        elif isinstance(model, Perceptron):
            return {
                "penalty": ["l2", "l1", "elasticnet"],
                "max_iter": [50, 100, 200]
            }
        elif isinstance(model, SGDClassifier):
            return {
                "alpha": np.logspace(-4, -1, 8),
                "max_iter": [100, 300, 500],
                "penalty": ["l2", "l1", "elasticnet"]
            }
        elif isinstance(model, SVC):
            return {
                "C": np.logspace(-1, 2, 15),
                "kernel": ["linear", "poly", "rbf", "sigmoid"]
            }
        elif isinstance(model, LinearDiscriminantAnalysis):
            return {
                "solver": ["svd", "lsqr", "eigen"],
                "shrinkage": [None, "auto"]
            }
        elif isinstance(model, PLSRegression):
            return {
                "n_components": [2, 3, 5]
            }
        elif isinstance(model, GaussianNB):
            return {
                "var_smoothing": np.logspace(-11, -8, 10)
            }
        elif isinstance(model, KNeighborsClassifier):
            return {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "p": [1, 2]
            }
        elif isinstance(model, GradientBoostingClassifier):
            return {
                "n_estimators": [50, 100, 200],
                "learning_rate": np.logspace(-4, -1, 10),
                "max_depth": [3, 5, 7]
            }
        else:
            return {}

    def train_and_score_model(self, model, X_train, X_test, y_train, y_test, seed):
        param_dist = self.get_tunable_params(model)

        rsh = GridSearchCV(estimator=model, param_grid=param_dist, cv=StratifiedKFold(3, shuffle=True, random_state=seed), scoring=self.tuneScoring, verbose=(self.debug > 3))

        rsh.fit(X_train, y_train)

        optimized_model = model.set_params(**rsh.best_params_)
        optimized_model.fit(X_train, y_train)

        y_pred = optimized_model.predict(X_test)

        # make a numpy array from y_pred where all the values > 0.5 become 1 and all remaining values are 0
        if type_of_target(y_pred) == "continuous":
            y_pred = np.where(y_pred > 0.5, 1, 0)


        brier = None
        if hasattr(optimized_model, "predict_proba"):
            try:
                y_prob = optimized_model.predict_proba(X_test)[:, 1]
                brier = brier_score_loss(y_test, y_prob)
            except Exception as e:
                pass

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        npv = tn / (tn + fn)
        ppv = tp / (tp + fp)

        self.trained += 1
        self.bar.update()
        return roc_auc, f1, accuracy, recall, sensitivity, specificity, npv, ppv, brier, optimized_model, json.dumps(rsh.best_params_)

    def run_dataset(self, label, X_train, X_test, y_train, y_test, seed, sublabel=None):
        node = os.uname()[1]
        for model in self.get_model_train():
            if sublabel is None:
                model_file = '{}/{}/{}_{}'.format(self.safe_name, label, type(model).__name__, seed )
                model_label = "{}".format(label)

            else:
                model_file = '{}/{}/{}_{}_{}'.format(self.safe_name, label, sublabel, type(model).__name__, seed )
                model_label = "{}-{}".format(label, sublabel)

            inEntry = ((self.ledger['Dataset']==model_label) & (self.ledger['Model']==type(model).__name__) & (self.ledger['Seed'] == seed)).any()

            if inEntry:
                if os.path.isfile(model_file):
                    continue
                else:
                    self.trained -= 1
                    self.ledger.drop(((self.ledger['Dataset']==model_label) & (self.ledger['Model']==type(model).__name__) & (self.ledger['Seed'] == seed)).index)

            roc_auc, f1, accuracy, recall, sensitivity, specificity, npv, ppv, brier, optimized_model, parms = self.train_and_score_model(model, X_train, X_test, y_train, y_test, seed)
            ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            joblib.dump(optimized_model, model_file)
            #[, , "Parms", "Seed", "Ratio", , , , "F1", , "ts", "node"]
            newrow = pd.DataFrame( [{"node": node,
                                "ts": ts,
                                "Dataset": model_label,
                                "Model": type(model).__name__,
                                "Params": parms,
                                "Seed": seed,
                                "Ratio": self.ratio,
                                "Accuracy": accuracy,
                                "Recall": recall,
                                "Sensitivity": sensitivity,
                                "Specificity": specificity,
                                "NPV": npv,
                                "PPV": ppv,
                                "Brier": brier,
                                "F1": f1,
                                "ROC_AUC": roc_auc,
                                }] )
            self.ledger = pd.concat([self.ledger, newrow], ignore_index=True)

        if self.dnn:
            if sublabel is None:
                model_file = '{}/{}/DNN_{}'.format(self.safe_name, label, seed )
                model_label = "{}".format(label)

            else:
                model_file = '{}/{}/{}_DNN_{}'.format(self.safe_name, label, sublabel, seed )
                model_label = "{}-{}".format(label, sublabel)

            inEntry = ((self.ledger['Dataset']==model_label) & (self.ledger['Model']=='DNN') & (self.ledger['Seed'] == seed)).any()

            if inEntry:
                if os.path.isfile(model_file):
                    return
                else:
                    self.trained -= 1
                    self.ledger.drop(((self.ledger['Dataset']==model_label) & (self.ledger['Model']=='DNN') & (self.ledger['Seed'] == seed)).index)

            roc_auc, f1, accuracy, recall, sensitivity, specificity, npv, ppv, brier, optimized_model, parms = self.train_and_score_model_keras(X_train, X_test, y_train, y_test, seed, model_label)
            ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
#            joblib.dump(optimized_model, model_file)
            #[, , "Parms", "Seed", "Ratio", , , , "F1", , "ts", "node"]
            newrow = pd.DataFrame( [{"node": node,
                                "ts": ts,
                                "Dataset": model_label,
                                "Model": 'DNN',
                                "Params": parms,
                                "Seed": seed,
                                "Ratio": self.ratio,
                                "Accuracy": accuracy,
                                "Recall": recall,
                                "Sensitivity": sensitivity,
                                "Specificity": specificity,
                                "NPV": npv,
                                "PPV": ppv,
                                "Brier": brier,
                                "F1": f1,
                                "ROC_AUC": roc_auc
                                }] )
            self.ledger = pd.concat([self.ledger, newrow], ignore_index=True)

    def fit(self):
        self.logger.info("{:=^60}".format(' Begin Fit {} Models '.format(self.total_models-self.trained)))
        manager = enlighten.get_manager()
        self.bar = manager.counter(total=self.total_models,
                                   count=self.trained,
                                   format='{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:{len_total}d}/{total:d} [{elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s]',
                                   desc='Tunning',
                                   unit='Models')

        for seed in self.seeds:
            X  = self.noMissingDataset.drop(self.name, axis=1)
            y = self.noMissingDataset[self.name]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.ratio, random_state=seed, stratify=y)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            joblib.dump(scaler, '{}/nomissing-original/StandardScaler_{}'.format(self.safe_name, seed) )

            self.run_dataset('nomissing-original', X_train_scaled, X_test_scaled, y_train, y_test, seed)

            if not self.is_balanced(y):
                ros = RandomOverSampler(random_state=seed)
                Xr_train_scaled, yr_train = ros.fit_resample(X_train_scaled, y_train)
                self.run_dataset('nomissing-oversampled', Xr_train_scaled, X_test_scaled, yr_train, y_test, seed)

            self.saveCheckPoint()

            for dfData, dfname in self.missingDatasets:
                mice = IterativeImputer(max_iter=10, random_state=seed)
                df_mice = dfData.copy()

                X  = df_mice.drop(self.name, axis=1)
                y = df_mice[self.name]
                X_mice = mice.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(X_mice, y, test_size=self.ratio, random_state=seed, stratify=y)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                joblib.dump(scaler, '{}/{}-original/mice_StandardScaler_{}'.format(self.safe_name, dfname, seed) )

                self.run_dataset('{}-original'.format(dfname), X_train_scaled, X_test_scaled, y_train, y_test, seed, 'mice')

                if not self.is_balanced(y):
                    ros = RandomOverSampler(random_state=seed)
                    Xr_train_scaled, yr_train = ros.fit_resample(X_train_scaled, y_train)
                    self.run_dataset('{}-oversampled'.format(dfname), Xr_train_scaled, X_test_scaled, yr_train, y_test, seed, 'mice')

                self.saveCheckPoint()

            for dfData, dfname in self.missingDatasets:
                knn = KNNImputer(n_neighbors=5)
                df_knn = dfData.copy()

                X  = df_knn.drop(self.name, axis=1)
                y = df_knn[self.name]
                X_knn = knn.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(X_knn, y, test_size=self.ratio, random_state=seed, stratify=y)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                joblib.dump(scaler, '{}/{}-original/knn_StandardScaler_{}'.format(self.safe_name, dfname, seed) )

                self.run_dataset('{}-original'.format(dfname), X_train_scaled, X_test_scaled, y_train, y_test, seed, 'knn')

                if not self.is_balanced(y):
                    ros = RandomOverSampler(random_state=seed)
                    Xr_train_scaled, yr_train = ros.fit_resample(X_train_scaled, y_train)
                    self.run_dataset('{}-oversampled'.format(dfname), Xr_train_scaled, X_test_scaled, yr_train, y_test, seed, 'knn')

                self.saveCheckPoint()
        self.bar.close()

    def get_best_models(self, metric="ROC_AUC"):
        return self.ledger.groupby(["Dataset", "Model"])[metric].agg(['mean', 'std'])

    def explain_model(self, modelname=None, dataset=None, seed=None):
        self.logger.info("{:=^60}".format(' Begin SHAP Explainer: {} {} {} '.format(modelname, dataset, seed)))

        Xbase = self.noMissingDataset.drop(self.name, axis=1)
        ybase = self.noMissingDataset[self.name]

        X_1 = self.noMissingDataset[ybase == 1].drop(self.name, axis=1)
        X_0 = self.noMissingDataset[ybase == 0].drop(self.name, axis=1)
        X_raw_explain = pd.concat([X_1[:5], X_0[:5]], ignore_index=True)

        self.logger.info("Model: {}".format(modelname))
        self.logger.info("Seed: {}".format(seed))
        pieces = dataset.split('-')

        dataset = pieces[0]
        sample = pieces[1]
        self.logger.info("Dataset: {}".format(dataset))
        self.logger.info("Sample: {}".format(sample))

        if pieces[-1] in (['mice', 'knn']):
            imputer = pieces[2]

            scaler_path = "{}/{}-original/{}_StandardScaler".format(self.safe_name,dataset, imputer)
            model_path = "{}/{}-{}/{}_{}".format(self.safe_name, dataset, sample, imputer, modelname)

            if dataset == 'fulldataset':
                X_na  = self.missingDatasets[0][0].drop(self.name, axis=1)
                y = self.missingDatasets[0][0][self.name]
            else:
                X_na  = self.missingDatasets[1][0].drop(self.name, axis=1)
                y = self.missingDatasets[1][0][self.name]

            if imputer == 'knn':
                knn = KNNImputer(n_neighbors=5)
                X = knn.fit_transform(X_na)

        else:
            imputer = None

            scaler_path = "{}/{}-original/StandardScaler".format(self.safe_name, '-'.join(pieces[:-1]))
            model_path = "{}/{}-{}/{}".format(self.safe_name, dataset, sample, modelname)

            X  = self.noMissingDataset.drop(self.name, axis=1)
            y = self.noMissingDataset[self.name]

        all_shap_base_values = []
        base_dim = []
        all_shap_values = []

        if imputer == 'mice':
            mice = IterativeImputer(max_iter=10, random_state=seed)
            X = mice.fit_transform(X_na)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=4, random_state=seed, stratify=y)

        scaler = joblib.load('{}_{}'.format(scaler_path, seed))
        model = joblib.load('{}_{}'.format(model_path, seed))

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_explain = scaler.transform(X_raw_explain)
        X_model = scaler.transform(Xbase)

        if not self.is_balanced(y):
            ros = RandomOverSampler(random_state=seed)
            X_train, y_train = ros.fit_resample(X_train, y_train)

#        explainer_model = shap.Explainer(model)

#        expected_value = explainer_model.expected_value
#        if isinstance(expected_value, list):
#            expected_value = expected_value[1]
#        shap_values = explainer.shap_values(X_test)[1]
        self.logger.info("Columns: {}".format(Xbase.columns))
#        label_columns = ['sex', 'family hist', 'age diag', 'BMI', 'base glu', 'glu 120','HbA1c']
        label_columns = ['sexo', 'hist fam', 'edad diag', 'IMC', 'glu ayu', 'glu 120','A1c']

        explainer = shap.Explainer(model.predict, X_train, seed=seed)
        shap_values = explainer(X_model)

        exp = shap.Explanation(shap_values,
                          data=X_model,
                          feature_names=label_columns)

        shap.plots.decision(exp.base_values[0], exp.values, features=label_columns, show=False)
        plt.title(r"Predicciones mejor modelo: {0}".format(modelname))
        plt.xlabel("Predicción del modelo: 0 Negativo, 1 Positivo")
        plt.savefig("{}/shap_{}_{}_{}.png".format(self.safe_name, modelname, dataset, seed),dpi=150, bbox_inches='tight')
        plt.close()

        y_pred = model.predict(X_model)
        # make a numpy array from y_pred where all the values > 0.5 become 1 and all remaining values are 0
        if type_of_target(y_pred) == "continuous":
            y_pred = np.where(y_pred > 0.5, 1, 0)

        X_pos = X_model[y_pred == 1]
        shap_values = explainer(X_pos)
        exp = shap.Explanation(shap_values,
                          data=X_pos,
                          feature_names=label_columns)

        shap.plots.decision(exp.base_values[0], exp.values, features=label_columns, show=False)
        plt.title(r"Predicciones mejor modelo: {0}=1".format(modelname))
        plt.xlabel("Predicción del modelo")
        plt.savefig("{}/shap_pos_{}_{}_{}.png".format(self.safe_name, modelname, dataset, seed),dpi=150, bbox_inches='tight')
        plt.close()


        X_pos = X_model[y_pred == 0]
        shap_values = explainer(X_pos)
        exp = shap.Explanation(shap_values,
                          data=X_pos,
                          feature_names=label_columns)

        shap.plots.decision(exp.base_values[0], exp.values, features=label_columns, show=False)
        plt.title(r"Predicciones mejor modelo: {0}=0".format(modelname))
        plt.xlabel("Predicción del modelo")
        plt.savefig("{}/shap_neg_{}_{}_{}.png".format(self.safe_name, modelname, dataset, seed),dpi=150, bbox_inches='tight')
        plt.close()


        shap_values = explainer(X_explain)

        exp = shap.Explanation(shap_values,
                          data=X_explain,
                          feature_names=label_columns)

        for i in range(5):
            shap.plots.waterfall(exp[i], show=False)
            plt.title(r"{0} $y_{{{1}}}=1$".format(modelname, i))
            plt.savefig("{}/pos_{}_{}_{}_{}.png".format(self.safe_name, i, modelname, dataset, seed),dpi=150, bbox_inches='tight')
            plt.close()

        for i in range(5, 10):
            shap.plots.waterfall(exp[i], show=False)
            plt.title(r"{0} $y_{{{1}}}=0$".format(modelname, i-5))
            plt.savefig("{}/neg_{}_{}_{}_{}.png".format(self.safe_name, i-5, modelname, dataset, seed),dpi=150, bbox_inches='tight')
            plt.close()

    def wrap_and_save(self):
        self.logger.info("{:=^60}".format(' Saving Summary and Wrap the output in a ZipFile '))
        for metric in ["ROC_AUC", "NPV", "PPV", "Brier", "Sensitivity", "Specificity"]:
            with pd.ExcelWriter('{}/Summary-{}.xlsx'.format(self.safe_name, metric) , engine='xlsxwriter') as xls:
                self.get_best_models(metric).to_excel(xls, sheet_name='Results')

        with zipfile.ZipFile('{}-{}.zip'.format(self.safe_name, self.start), 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.safe_name):
                for file in files:
                    zipf.write(os.path.join(root, file))
