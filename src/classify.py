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
SHORE_FILE_PATH = '../results/shore_mean.csv'
  

def split_data(exp=1):
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
    #acou_feat_df = acou_feat_df[['naq','frequency','energy']]
    acou_feat_df = acou_feat_df.replace([np.inf, -np.inf], np.nan)
    acou_feat_df = acou_feat_df.apply(lambda x: x.fillna(x.mean()),axis=0)

    visual_feat_df = pd.read_csv(VISUAL_FILE_PATH)
    #visual_feat_df = visual_feat_df[['face_up_down','mouth_open','smile_level']]

    shore_feat_df = pd.read_csv(SHORE_FILE_PATH)
    shore_feat_df = shore_feat_df[['Sad', 'LeftEyeClosed', 'Happy']]

    pd.concat([acou_feat_df, visual_feat_df, shore_feat_df, labels_df], axis=1)

    full_data = pd.concat([ labels_df, acou_feat_df, shore_feat_df], axis=1)
    full_data.to_csv("../results/full_data.csv", index=False)

    return pd.concat([acou_feat_df, shore_feat_df, labels_df], axis=1)
    #return pd.concat([acou_feat_df, visual_feat_df, shore_feat_df, labels_df], axis=1)

def get_values(training_set,validation_set,test_set):
    
    #find value of C

    training_data = training_set[:,:-1]
    training_labels = training_set[:,-1]
    validation_data = validation_set[:,:-1]
    validation_labels = validation_set[:,-1]
    testing_data = test_set[:,:-1]
    testing_labels = test_set[:,-1]

    print "train:",training_data.shape
    print "val:",validation_data.shape
    print "test:",testing_data.shape

    params = {'C': [0.001,0.01,1,10,100,1000]}
    data = np.concatenate((training_data, validation_data), axis=0)
    print "data:",data.shape
    labels = np.concatenate([training_labels, validation_labels])
    train_indices = range(len(training_data))
    validation_indices = range(len(training_data),len(validation_data)+len(training_data))

    #clf_find_c = svm.SVC(decision_function_shape='ovr',kernel='rbf')
    clf_find_c = svm.LinearSVC(dual=True, penalty='l2')
    grid_clf = GridSearchCV(estimator=clf_find_c, n_jobs=-1, param_grid=params,cv=[(train_indices,validation_indices)])
    grid_clf.fit(data,labels)
    c = grid_clf.best_estimator_.C
    print c

    #train accuracy
    #best_clf = svm.SVC(C=c,decision_function_shape='ovr',kernel='linear')
    best_clf = svm.LinearSVC(C=c, dual=True)
    best_clf.fit(data,labels)
    training_accuracy = best_clf.score(data,labels)
    print "training accuracy:", training_accuracy

    #validation accuracy

    validation_accuracy = best_clf.score(validation_data,validation_labels)
    print "validation accuracy:", validation_accuracy

    #test accuracy

    testing_accuracy = best_clf.score(testing_data,testing_labels)
    print "testing accuracy:", testing_accuracy


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
