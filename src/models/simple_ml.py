import os
from fnmatch import fnmatch

import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def get_statistics(array):
    statistics = np.concatenate([[np.mean(array, axis=1)],
                                 [np.var(array, axis=1)],
                                 [np.mean(array, axis=1) * np.var(array, axis=1)],
                                 [np.mean(array, axis=1) ** 2],
                                 [np.var(array, axis=1) ** 2],
                                 [np.quantile(array, 0.75, axis=1)],
                                 [np.quantile(array, 0.25, axis=1)]], axis=0)
    statistics = statistics * 100000
    return statistics.T


def getListOfFiles(dirName, pattern="*.jpg"):
    allFiles = []
    for path, subdirs, files in os.walk(dirName):
        for name in files:
            if pattern != "":
                if fnmatch(name, pattern):
                    allFiles.append(os.path.join(path, name))
            else:
                allFiles.append(os.path.join(path, name))
    return allFiles


def csv_folder_paers(csv_folder):
    csv_files = getListOfFiles(csv_folder, "*.csv")
    X_train, X_test, y_train, y_test = [], [], [], []
    for csv_file in csv_files:
        contains = np.loadtxt(csv_file, delimiter=';')
        type = 1 if 'true' in csv_file else 0
        if 'train' in csv_file:
            X_train.append(contains)
            y_train.append(np.repeat(type, contains.shape[0]))
        if 'test' in csv_file:
            X_test.append(contains)
            y_test.append(np.repeat(type, contains.shape[0]))
    return [np.concatenate(i, axis=0) for i in [X_train, X_test, y_train, y_test]]


def train_save_model():
    # fakes1=np.loadtxt('fake_persons.csv',delimiter=';')
    # fakes2=np.loadtxt('fake_persons_2.csv',delimiter=';')
    # fakes3=np.loadtxt('fake_persons_3.csv',delimiter=';')
    #
    # fakes=np.concatenate([fakes1,fakes2,fakes3])
    #
    # trues=np.loadtxt('true_persons.csv',delimiter=';')

    # # fakes=np.concatenate([get_statistics(fakes),fakes],axis=1)
    # # trues=np.concatenate([get_statistics(trues),trues],axis=1)
    # X=np.concatenate([trues,fakes],axis=0)
    #
    # y=np.repeat(1,trues.shape[0]).tolist()+np.repeat(0,fakes.shape[0]).tolist()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_train, X_test, y_train, y_test = csv_folder_paers('/home/maksym/Documents/proj_3d/ds/csv_ds')
    # X_test=get_statistics(X_test)
    # X_train=get_statistics(X_train)
    models = [RandomForestClassifier(n_estimators=100), LogisticRegression(), svm.SVC(probability=True)]
    [model.fit(X_train, y_train) for model in models]
    [model.score(X_test, y_test) for model in models]
    print([roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) for model in models])

    from joblib import dump, load
    dump(models, 'src/models/models9.joblib')
    clfs = load('src/models/models9.joblib')
    print([roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]) for clf in clfs])
