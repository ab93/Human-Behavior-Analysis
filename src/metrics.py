import os
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

RESULTS_PATH = '../results/scores'

def generate_confusion_matrix(name, *args):
    for i in range(len(args)):
        values = np.loadtxt(os.path.join(RESULTS_PATH,args[i]), delimiter='\t')
        if i == 0:
            data = values
        else:
            data = np.concatenate((data,values),axis=0)

    def plot_confusion_matrix(data, info):
        y_true, y_pred = data[:,0], data[:,1]
        cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
        
        classes = [-1,0,1]
        plt.clf()
        plt.close()
        fig = plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Normalized Confusion matrix " + '(' + info + ')')
        plt.colorbar()
        tick_marks = np.arange(3)
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes)

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #np.set_printoptions(precision=2, suppress=True)
        cm = np.around(cm, decimals=3)
        print cm

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        #plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        #plt.show()
        fig.savefig('../plots/metrics/' + info + '.png')


    plot_confusion_matrix(data, name)

if __name__ == '__main__':
    generate_confusion_matrix('Naive Bayes' ,'NB1.txt','NB2.txt','NB3.txt','NB4.txt')
    generate_confusion_matrix('SVM RBF' ,'RBF1.txt','RBF2.txt','RBF3.txt','RBF4.txt')
    generate_confusion_matrix('Random Forests' ,'RF1.txt','RF2.txt','RF3.txt','RF4.txt')
    generate_confusion_matrix('SVM Linear' ,'SVM_linear_hold_Exp_1_1.txt',
                            'SVM_linear_hold_Exp_1_2.txt','SVM_linear_hold_Exp_1_3.txt',
                            'SVM_linear_hold_Exp_1_3.txt')

