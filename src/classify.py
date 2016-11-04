import numpy as np
import pandas as pd
from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import random

ANNOT_FILE_PATH = '../data/sentimentAnnotations_rev_v03.xlsx'
ACOUSTIC_FILE_PATH = '../results/acou_mean.csv'
VISUAL_FILE_PATH = '../results/okao_mean.csv'
SHORE_FILE_PATH = '../results/shore_mean.csv'
  
def plot_validation_curve(X, y, clf, cv, param_name, param_range, title=''):
    train_scores, test_scores = validation_curve(clf, X, y, 
                                param_name=param_name, param_range=param_range,
                                cv=cv, n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig = plt.figure()
    plt.title("Validation Curve (Linear SVM) " + title)
    plt.xlabel("C")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2,
                    color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2,
                    color="navy", lw=lw)
    plt.legend(loc="best")
    #plt.show()
    fig.savefig('../plots/val_curve/VC_'+title+'.png')

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
    acou_feat_df = acou_feat_df.drop('polarity',1)
    acou_feat_df = acou_feat_df[['naq','v_u','energy']]
    #acou_feat_df = acou_feat_df[['naq']]
    acou_feat_df = acou_feat_df.replace([np.inf, -np.inf], np.nan)
    acou_feat_df = acou_feat_df.apply(lambda x: x.fillna(x.mean()),axis=0)

    visual_feat_df = pd.read_csv(VISUAL_FILE_PATH)
    #visual_feat_df = visual_feat_df[['face_up_down','mouth_open','smile_level']]
    visual_feat_df = visual_feat_df[['mouth_open']]

    shore_feat_df = pd.read_csv(SHORE_FILE_PATH)
    shore_feat_df = shore_feat_df.drop('Video',1)
    shore_feat_df = shore_feat_df[['LeftEyeClosed', 'Angry']]
    #shore_feat_df = shore_feat_df[['LeftEyeClosed', 'Happy']]

    #full_data = pd.concat([acou_feat_df, shore_feat_df, visual_feat_df, labels_df], axis=1)
    #full_data.to_csv("../results/full_data.csv", index=False)

    return pd.concat([acou_feat_df, visual_feat_df, shore_feat_df, labels_df], axis=1)
    #return pd.concat([acou_feat_df, visual_feat_df, shore_feat_df, labels_df], axis=1)

def get_acou_features():

    labels_df = pd.read_excel(ANNOT_FILE_PATH, sheetname='Sheet1')
    labels_df = labels_df['majority vote']

    acou_feat_df = pd.read_csv(ACOUSTIC_FILE_PATH)
    acou_feat_df = acou_feat_df.drop('polarity', 1)
    acou_feat_df = acou_feat_df[['naq', 'v_u', 'energy']]
    # acou_feat_df = acou_feat_df[['naq']]
    acou_feat_df = acou_feat_df.replace([np.inf, -np.inf], np.nan)
    acou_feat_df = acou_feat_df.apply(lambda x: x.fillna(x.mean()), axis=0)
    return pd.concat([acou_feat_df,labels_df],axis = 1)

def get_visual_featues():

    labels_df = pd.read_excel(ANNOT_FILE_PATH, sheetname='Sheet1')
    labels_df = labels_df['majority vote']

    visual_feat_df = pd.read_csv(VISUAL_FILE_PATH)
    #visual_feat_df = visual_feat_df[['face_up_down','mouth_open','smile_level']]
    visual_feat_df = visual_feat_df[['mouth_open']]

    shore_feat_df = pd.read_csv(SHORE_FILE_PATH)
    shore_feat_df = shore_feat_df.drop('Video', 1)
    shore_feat_df = shore_feat_df[['LeftEyeClosed', 'Angry']]
    # shore_feat_df = shore_feat_df[['LeftEyeClosed', 'Happy']]

    return pd.concat([visual_feat_df, shore_feat_df, labels_df], axis=1)


def get_values(training_set,validation_set,test_set, exp_no, val='hold'):
    
    # find value of C

    training_data = training_set[:,:-1]
    training_labels = training_set[:,-1]
    validation_data = validation_set[:,:-1]
    validation_labels = validation_set[:,-1]
    testing_data = test_set[:,:-1]
    testing_labels = test_set[:,-1]

    # print "train:",training_data.shape
    # print "val:",validation_data.shape
    # print "test:",testing_data.shape

    params = {'C': [0.001,0.01,1,10,100,1000]}
    data = np.concatenate((training_data, validation_data), axis=0)
    #print "data:",data.shape
    labels = np.concatenate([training_labels, validation_labels])
    train_indices = range(len(training_data))
    validation_indices = range(len(training_data),len(validation_data)+len(training_data))

    #clf_find_c = svm.SVC(decision_function_shape='ovr',kernel='rbf')
    clf_find_c = svm.LinearSVC(dual=False, penalty='l2')
    
    if val == "hold":
        grid_clf = GridSearchCV(estimator=clf_find_c, n_jobs=-1, param_grid=params,cv=[(train_indices,validation_indices)])
        plot_validation_curve(data, labels, svm.LinearSVC(dual=False), cv=[(train_indices,validation_indices)], 
                            param_name="C", param_range=params['C'], title=('hold' + exp_no))
    else:
        grid_clf = GridSearchCV(estimator=clf_find_c, n_jobs=-1, param_grid=params,cv=3)
        plot_validation_curve(data, labels, svm.LinearSVC(dual=False), cv=3, 
                            param_name="C", param_range=params['C'], title=('3-fold' + exp_no))
    
    
    grid_clf.fit(data,labels)
    c = grid_clf.best_estimator_.C
    print "C:",c

    #train accuracy
    #best_clf = svm.SVC(C=c,decision_function_shape='ovr',kernel='linear')
    best_clf = svm.LinearSVC(C=c, dual=False)
    best_clf.fit(data,labels)
    training_accuracy = best_clf.score(data,labels)
   # print "training accuracy:", training_accuracy

    #validation accuracy

    validation_accuracy = best_clf.score(validation_data,validation_labels)
    #print "validation accuracy:", validation_accuracy

    #test accuracy

    testing_accuracy = best_clf.score(testing_data,testing_labels)
    print "testing accuracy:", testing_accuracy

    return testing_accuracy

def model(exp):
    temp = []
    if exp == "Exp 1":
        df = fuse_features()
    elif exp == "Exp 2 a":
        df = get_acou_features()
    elif exp == "Exp 2 b":
        df = get_visual_featues()

    for i in range(4):
        print "\nExperiment " + str(i) 
        train_idx, val_idx, test_idx = split_data(i)

        train_features = df.iloc[train_idx,:-1]
        train_labels = df.iloc[train_idx,-1]
        training_set = pd.concat([train_features, train_labels], axis=1)

        val_features = df.iloc[val_idx,:-1]
        val_labels = df.iloc[val_idx,-1]
        validaition_set = pd.concat([val_features, val_labels], axis=1)

        test_features = df.iloc[test_idx,:-1]
        test_labels = df.iloc[test_idx,-1]
        test_set = pd.concat([test_features, test_labels], axis=1)

        print "\nHold-out:"
        temp.append(get_values(training_set.values, validaition_set.values, test_set.values, str(i+1), val='hold' ))
        if exp == "Exp 1":
            print "\n3-Fold:"
            get_values(training_set.values, validaition_set.values, test_set.values, str(i+1), val='3-fold')

    return temp


def draw_multi_plots(acou_accuracy_list,visual_accuracy_list):
    print acou_accuracy_list
    print visual_accuracy_list
    plt.plot([1,2,3,4],acou_accuracy_list, 'r-')
    plt.plot([1,2,3,4],visual_accuracy_list,'b-')
    plt.axis([0, 5, 0,1])
    plt.show()


if __name__ == '__main__':
    model("Exp 1")
    #acoustic = model("Exp 2 a")
    #visual = model("Exp 2 b")
    #draw_multi_plots(acoustic,visual)

