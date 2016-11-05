import numpy as np
import pandas as pd
from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import random
import scipy
from sklearn.naive_bayes import GaussianNB

ANNOT_FILE_PATH = '../data/sentimentAnnotations_rev_v03.xlsx'
ACOUSTIC_FILE_PATH = '../results/acou_mean.csv'
VISUAL_FILE_PATH = '../results/okao_mean.csv'
SHORE_FILE_PATH = '../results/shore_mean.csv'
split = [None] * 4
  
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

def randomize():
    df = pd.read_excel(ANNOT_FILE_PATH, sheetname='Sheet1')
    grouped = df.groupby('video')
    data = grouped.groups.items()
    random.shuffle(data)

    split[0] = [item for sublist in [tuple_[1] for tuple_ in data[:12]] for item in sublist]
    split[1] = [item for sublist in [tuple_[1] for tuple_ in data[12:24]] for item in sublist]
    split[2] = [item for sublist in [tuple_[1] for tuple_ in data[24:36]] for item in sublist]
    split[3] = [item for sublist in [tuple_[1] for tuple_ in data[36:48]] for item in sublist]

def split_data(exp=1):
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

def get_values(training_set,validation_set,test_set, exp_no, val='hold', VC=True):
    
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
        if VC:
            plot_validation_curve(data, labels, svm.LinearSVC(dual=False), cv=[(train_indices,validation_indices)], 
                            param_name="C", param_range=params['C'], title=('hold' + exp_no))
    else:
        grid_clf = GridSearchCV(estimator=clf_find_c, n_jobs=-1, param_grid=params,cv=3)
        if VC:
            plot_validation_curve(data, labels, svm.LinearSVC(dual=False), cv=3, 
                            param_name="C", param_range=params['C'], title=('3-fold' + exp_no))
    
    
    grid_clf.fit(data,labels)
    validation_accuracy = grid_clf.best_score_

    c = grid_clf.best_estimator_.C
    print "C:",c
    print "validation_accuracy:", validation_accuracy
    #train accuracy
    
    best_clf = svm.LinearSVC(C=c, dual=False)
    best_clf.fit(data,labels)
    #best_clf.fit(training_data, training_labels)
    training_accuracy = best_clf.score(data,labels)
    print "training accuracy:", training_accuracy

    #test accuracy
    
    testing_accuracy = best_clf.score(testing_data,testing_labels)
    print "testing accuracy:", testing_accuracy
    test_predictions = best_clf.predict(testing_data)
    accuracies = [training_accuracy, validation_accuracy, testing_accuracy]
    return accuracies, test_predictions

def get_svm_rbf_values(training_set,validation_set,test_set):
    training_data = training_set[:,:-1]
    training_labels = training_set[:,-1]
    validation_data = validation_set[:,:-1]
    validation_labels = validation_set[:,-1]
    testing_data = test_set[:,:-1]
    testing_labels = test_set[:,-1]

    param_grid = {'C':[0.001,0.01,1,10,100,1000], 'gamma': np.logspace(-3,3,6)}

    data = np.concatenate((training_data, validation_data), axis=0)
    #print "data:",data.shape
    labels = np.concatenate([training_labels, validation_labels])
    train_indices = range(len(training_data))
    validation_indices = range(len(training_data),len(validation_data)+len(training_data))

    clf_rf = svm.SVC()

    grid_clf = GridSearchCV(estimator=clf_rf, n_jobs=-1, param_grid=param_grid,cv=[(train_indices,validation_indices)])

    grid_clf.fit(data,labels)
    print "\nSVM RBF:"
    print grid_clf.best_params_
    validation_accuracy = grid_clf.best_score_
    print "validation_accuracy:", validation_accuracy

    #train accuracy

    best_clf = grid_clf.best_estimator_
    best_clf.fit(data,labels)
    training_accuracy = best_clf.score(data,labels)
    print "training accuracy:", training_accuracy

    #test accuracy
    testing_accuracy = best_clf.score(testing_data,testing_labels)
    print "testing accuracy:", testing_accuracy
    return (training_accuracy, validation_accuracy, testing_accuracy)

def get_rf_values(training_set,validation_set,test_set):
    training_data = training_set[:,:-1]
    training_labels = training_set[:,-1]
    validation_data = validation_set[:,:-1]
    validation_labels = validation_set[:,-1]
    testing_data = test_set[:,:-1]
    testing_labels = test_set[:,-1]

    param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 6],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "criterion": ["gini", "entropy"]}

    data = np.concatenate((training_data, validation_data), axis=0)
    #print "data:",data.shape
    labels = np.concatenate([training_labels, validation_labels])
    train_indices = range(len(training_data))
    validation_indices = range(len(training_data),len(validation_data)+len(training_data))

    clf_rf = RandomForestClassifier()

    grid_clf = GridSearchCV(estimator=clf_rf, n_jobs=-1, param_grid=param_grid,cv=[(train_indices,validation_indices)])

    grid_clf.fit(data,labels)
    print "\nRandom Forests:"
    print grid_clf.best_params_
    validation_accuracy = grid_clf.best_score_
    print "validation_accuracy:", validation_accuracy

    #train accuracy

    best_clf = grid_clf.best_estimator_
    best_clf.fit(data,labels)
    training_accuracy = best_clf.score(data,labels)
    print "training accuracy:", training_accuracy

    #test accuracy
    testing_accuracy = best_clf.score(testing_data,testing_labels)
    print "testing accuracy:", testing_accuracy
    return (training_accuracy, validation_accuracy, testing_accuracy)

def plot_val_test(scores, title, filename):
    val_scores = [score[1] for score in scores]
    test_scores = [score[2] for score in scores]
    plt.clf()
    plt.close()
    fig = plt.figure()
    plt.title(title)
    plt.xlabel("Test Folds ID")
    plt.ylabel("Accuracy")
    plt.ylim(0.0,1.1)
    plt.xlim(0,5)
    plt.xticks(np.arange(0,6))
    plt.plot(np.arange(1,5), val_scores, 'o-', label="Validation Scores", color='red')
    plt.plot(np.arange(1,5), test_scores, 'o-', label="Test Scores", color='blue')
    plt.legend(loc='best')
    fig.savefig('../plots/val_test/' + filename + '.png')

def exp3_plot(lsvm_accuracies,gnb_accuracies,rbf_accuracies,rf_accuracies):
    lsvm_accuracies = [score[2] for score in lsvm_accuracies]
    rbf_accuracies = [score[2] for score in rbf_accuracies]
    rf_accuracies = [score[2] for score in rf_accuracies]
    plt.clf()
    plt.close()
    fig = plt.figure()
    print gnb_accuracies
    plt.title("Multiple classifiers")
    plt.xlabel("Test Folds ID")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.1)
    plt.xlim(0, 5)
    plt.xticks(np.arange(0, 6))
    plt.plot(np.arange(1, 5), lsvm_accuracies, 'o-', label="Linear SVM", color='red')
    plt.plot(np.arange(1, 5), gnb_accuracies, 'o-', label="Gaussian Naive Bayes", color='blue')
    plt.plot(np.arange(1, 5), rbf_accuracies, 'o-', label="SVM RBF kernel", color='green')
    plt.plot(np.arange(1, 5), rf_accuracies, 'o-', label="Random Forest", color='orange')
    plt.legend(loc='best')
    fig.savefig('../plots/val_test/' + "multi_classifier" + '.png')


def draw_multi_plots(multi_accuracy_list,acou_accuracy_list,visual_accuracy_list):

    acou_val_scores = [score[1] for score in acou_accuracy_list]
    acou_test_scores = [score[2] for score in acou_accuracy_list]
    visual_val_scores = [score[1] for score in visual_accuracy_list]
    visual_test_scores = [score[2] for score in visual_accuracy_list]
    multi_val_scores = [score[1] for score in multi_accuracy_list]
    multi_test_scores = [score[2] for score in multi_accuracy_list]

    plt.clf()
    plt.close()
    fig = plt.figure()
    plt.title("Acoustic Only")
    plt.xlabel("Test Folds ID")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.1)
    plt.xlim(0, 5)
    plt.xticks(np.arange(0, 6))
    plt.plot(np.arange(1, 5), acou_val_scores, 'o-', label="Acoustic Validation Scores", color='red')
    plt.plot(np.arange(1, 5), acou_test_scores, 'o-', label="Acoustic Test Scores", color='blue')
    plt.legend(loc='best')
    #plt.axis([0, 5, 0,1])
    fig.savefig('../plots/val_test/' + "acoustic_only" + '.png')

    plt.clf()
    plt.close()
    fig = plt.figure()
    plt.title("Visual Only")
    plt.xlabel("Test Folds ID")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.1)
    plt.xlim(0, 5)
    plt.xticks(np.arange(0, 6))
    plt.plot(np.arange(1, 5), visual_val_scores, 'o-', label="Visual Validation Scores", color='red')
    plt.plot(np.arange(1, 5), visual_test_scores, 'o-', label="Visual Test Scores", color='blue')
    plt.legend(loc='best')
    # plt.axis([0, 5, 0,1])
    fig.savefig('../plots/val_test/' + "visual_only" + '.png')

    plt.clf()
    plt.close()
    fig = plt.figure()
    plt.title("Acoustic vs Multimodal")
    plt.xlabel("Test Folds ID")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.1)
    plt.xlim(0, 5)
    plt.xticks(np.arange(0, 6))
    plt.plot(np.arange(1, 5), acou_test_scores, 'o-', label="Acoustic Test Scores", color='red')
    plt.plot(np.arange(1, 5), multi_test_scores, 'o-', label="Multimodal Test Scores", color='blue')
    plt.legend(loc='best')
    # plt.axis([0, 5, 0,1])

    fig.savefig('../plots/val_test/' + "acoustic+multi" + '.png')

    plt.clf()
    plt.close()
    fig = plt.figure()
    plt.title("Visual vs Multimodal")
    plt.xlabel("Test Folds ID")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.1)
    plt.xlim(0, 5)
    plt.xticks(np.arange(0, 6))
    plt.plot(np.arange(1, 5), visual_test_scores, 'o-', label="Visual Test Scores", color='red')
    plt.plot(np.arange(1, 5), multi_test_scores, 'o-', label="Multimodal Test Scores", color='blue')
    plt.legend(loc='best')
    # plt.axis([0, 5, 0,1])
    fig.savefig('../plots/val_test/' + "visual+multi" + '.png')

    plt.clf()
    plt.close()
    fig = plt.figure()
    plt.title("Visual vs Acoustic")
    plt.xlabel("Test Folds ID")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.1)
    plt.xlim(0, 5)
    plt.xticks(np.arange(0, 6))
    plt.plot(np.arange(1, 5), visual_test_scores, 'o-', label="Visual Test Scores", color='red')
    plt.plot(np.arange(1, 5), acou_test_scores, 'o-', label="Acoustic Test Scores", color='blue')
    plt.legend(loc='best')
    # plt.axis([0, 5, 0,1])
    fig.savefig('../plots/val_test/' + "visual+acoustic" + '.png')

def Gaussian_NB(training_set,test_set):
    training_data = training_set[:, :-1]
    training_labels = training_set[:, -1]
    testing_data = test_set[:, :-1]
    testing_labels = test_set[:, -1]
    clf = GaussianNB()
    clf.fit(training_data, training_labels)
    testing_accuracy = clf.score(testing_data,testing_labels)
    test_predictions = clf.predict(testing_data)

    return testing_accuracy,test_predictions




def model(exp):
    hold_out_scores = []
    fold_scores = []
    rbf_scores = []
    rf_scores = []
    predictions = []
    gaussian_values = []
    gaussian_predictions = []

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
        a,p = get_values(training_set.values, validaition_set.values, test_set.values,
                                str(i+1), val='hold')
        hold_out_scores.append(a)
        predictions.append(p)
        if exp == "Exp 1":
            print "\n3-Fold:"
            a, p = get_values(training_set.values, validaition_set.values, test_set.values, str(i + 1), val='3-fold')
            fold_scores.append(a)

            temp_a,temp_p = Gaussian_NB(training_set.values,test_set.values)
            gaussian_values.append(temp_a)
            gaussian_predictions.append(temp_p)


            rf_scores.append(get_rf_values(training_set.values, validaition_set.values, test_set.values))
            rbf_scores.append(get_svm_rbf_values(training_set.values, validaition_set.values, test_set.values))

    if exp == "Exp 1":
        plot_val_test(fold_scores, "3 fold scores", "3_fold")
        print gaussian_values
        print t_test_values(np.concatenate(hold_out_scores).ravel().tolist(),
                      np.concatenate(gaussian_predictions).ravel().tolist())
        exp3_plot(hold_out_scores,gaussian_values,rbf_scores,rf_scores)




    return hold_out_scores,predictions

def t_test_values(prediction_1,prediction_2):
    t_value = scipy.stats.ttest_ind(prediction_1,prediction_2,None ,True)
    return t_value

if __name__ == '__main__':
    randomize()
    print "Experiment A"
    multi_accuracies,multimodal_predictions = model("Exp 1")
    plot_val_test(multi_accuracies, "Hold out scores", "hold_out")
    print "Experiment 2 a"
    acoustic_accuracies,acoustic_predictions = model("Exp 2 a")
    print "Experiment 2 b"
    visual_accuracies,visual_predictions = model("Exp 2 b")
    draw_multi_plots(multi_accuracies,acoustic_accuracies,visual_accuracies)
    print "t scores"
    print t_test_values(np.concatenate(multimodal_predictions).ravel().tolist(), np.concatenate(acoustic_predictions).ravel().tolist())
    print t_test_values(np.concatenate(multimodal_predictions).ravel().tolist(),np.concatenate(visual_predictions).ravel().tolist())
    print t_test_values(np.concatenate(acoustic_predictions).ravel().tolist(),np.concatenate(visual_predictions).ravel().tolist())
