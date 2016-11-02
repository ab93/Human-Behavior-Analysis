import numpy as np
import pandas as pd
from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import random

ANNOT_FILE_PATH = '../data/sentimentAnnotations_rev_v03.xlsx'
ACOUSTIC_FILE_PATH = '../results/acou_mean.csv'
VISUAL_FILE_PATH = '../results/okao_mean.csv'
  

def split_data(exp=0):
    df = pd.read_excel(ANNOT_FILE_PATH, sheetname='Sheet1')
    grouped = df.groupby('video')
    data = grouped.groups.items()
    random.shuffle(data)
    split = [None] * 4
    split[0] = [ item for sublist in [tuple_[1] for tuple_ in data[:12]] for item in sublist]
    split[1] = [ item for sublist in [tuple_[1] for tuple_ in data[12:24]] for item in sublist]
    split[2] = [ item for sublist in [tuple_[1] for tuple_ in data[24:36]] for item in sublist]
    split[3] = [ item for sublist in [tuple_[1] for tuple_ in data[36:48]] for item in sublist]

    if exp == 0:
        return np.append(split[0],split[1]), split[2], split[3]
    elif exp == 1:
        return np.append(split[0],split[3]), split[1], split[2]
    elif exp == 2:
        return np.append(split[2],split[3]), split[0], split[1]
    else:
        return np.append(split[1],split[2]), split[3], split[0]
    

def fuse_features():
    labels_df = pd.read_excel(ANNOT_FILE_PATH, sheetname='Sheet1')
    labels_df = labels_df['majority vote']

    acou_feat_df = pd.read_csv(ACOUSTIC_FILE_PATH)
    acou_feat_df = acou_feat_df[['naq','frequency','energy']]

    acou_feat_df = acou_feat_df.replace([np.inf, -np.inf], np.nan)
    acou_feat_df = acou_feat_df.apply(lambda x: x.fillna(x.mean()),axis=0)

    visual_feat_df = pd.read_csv(VISUAL_FILE_PATH)
    visual_feat_df = visual_feat_df[['face_up_down','mouth_open','smile_level']]

    return pd.concat([acou_feat_df, visual_feat_df, labels_df], axis=1)

def get_values(training_set,validation_set,test_set):
    
    #validation accuracy
    training_data = training_set[:,:-1]
    training_labels = training_set[:,-1]
    validation_data = validation_set[:,:-1]
    validation_labels = validation_set[:,-1]
    params = {'C': [0.001,0.01,1,10,100]}
    data = np.concatenate((training_data, validation_data), axis=0)
    labels = np.concatenate([training_labels, validation_labels])
    train_indices = range(len(training_data))
    validation_indices = range(len(training_data),len(validation_data)+len(training_data))
    print train_indices, validation_indices
    print len(training_data),len(validation_data)
    print len(data),len(labels)
    print data.shape,labels.shape
    clf_validation = svm.SVC(C=1,decision_function_shape='ovo',kernel='linear')
    grid_clf = GridSearchCV(estimator=clf_validation, param_grid=params,cv=zip(train_indices,validation_indices))
    grid_clf.fit(data,labels)
    c = grid_clf.best_estimator_.C


    '''
    #training accuracy
    clf_training = svm.SVC(decision_function_shape='ovr',kernel='linear',C=c)
    clf_training.fit(training_set[:,:-1], training_set[:,-1])
    training_accuracy = clf_training.score(training_set[:,:-1], training_set[:,-1])
    print training_accuracy

    #test accuracy
    testing_accuracy = clf_training.score(test_set[:,:-1],test_set[:,-1])
    print testing_accuracy
    '''

def model():
    df = fuse_features()
    train_idx, val_idx, test_idx = split_data()

    train_features = df.iloc[train_idx,:-1]
    train_labels = df.iloc[train_idx,-1]
    training_set = pd.concat([train_features, train_labels], axis=1)

    val_features = df.iloc[val_idx,:-1]
    val_labels = df.iloc[val_idx,-1]
    validaition_set = pd.concat([val_features, val_labels], axis=1)

    test_features = df.iloc[test_idx,:-1]
    test_labels = df.iloc[test_idx,-1]
    test_set = pd.concat([test_features, test_labels], axis=1)

    get_values(training_set.values, validaition_set.values, test_set.values)

    #scaler = preprocessing.StandardScaler().fit(train_features)
    #x_train = scaler.transform(train_features)
    #x_val = scaler.transform(val_features)
    #print train_features[:,0].shape
    #raw_input()

    


    '''
    plt.figure(2, figsize=(8, 6))
    plt.scatter(train_features[:, 0], train_features[:, 1], c=train_labels, cmap=plt.cm.Paired)
    plt.xlabel('naq')
    plt.ylabel('mouth_open')
    

    #plt.xlim(x_min, x_max)
    #plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    #plt.show()
    

    clf = svm.SVC(C=1)
    #clf = svm.LinearSVC(C=0.1, dual=False, multi_class='ovr',)
    #clf = RandomForestClassifier()
    clf.fit(train_features, train_labels)
    print clf.predict(val_features)
    print val_labels
    #print train_labels

    print clf.score(val_features, val_labels)
    '''






if __name__ == '__main__':
    model()
