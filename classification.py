import pandas as pd
import numpy as np
import scipy.stats as ss
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv
from numpy import genfromtxt
from scipy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import preprocessing
import csv
from sklearn import linear_model
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix as c_m
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier as RDF
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import warnings
warnings.filterwarnings("ignore")
import time
from matplotlib.colors import ListedColormap
from sklearn import neighbors




class Lda_alpha(object):
    '''
    Classifications in Generalized Convex Spaces
    
    Attributes
    ---------
        alpha : float

    methods
    --------
        phi_alpha()
        phi_alpha_matrix()
    '''

    def __init__(self, alpha):
        
        self.alpha = alpha

    def phi_alpha(self, a):
        '''
        Define the function phi_alpha^-1
        :param alpha (a) (float)
        '''
        if a < 0:
            return -np.abs(a)**(1/self.alpha)
        elif a == 0:
            return 0
        else:
            return a**(1/self.alpha)
        
    def phi_alpha_matrix(self, x, a):
        '''
        Define the function phi_alpha^-1 for matrix
        :param x (document .xlsx)
        :param alpha (a) (float != 0)
        :out B (array)
        '''
        n, k = x.shape
        B = np.zeros_like(x)
        for j in range(k):
            for i in range(n):
                if x[i,j] < 0:
                    B[i,j] = -np.abs(x[i,j])**(1/a)
                elif x[i,j] == 0:
                    B[i,j] = 0
                else:
                    B[i,j] = x[i,j]**(1/a)
        return B



#Define the iris database
iris = load_iris()
x = iris.data
n, k = x.shape
y = iris.target
nb_groups = 10
nn = np.zeros((nb_groups, 1)) #Size of groups

for i in range(nb_groups):
    nn[i,:] = (y == i).sum()    
nn_cumul = np.cumsum(nn)
nn_cumul = np.r_[0,nn_cumul]
print(nn_cumul) #size of each group

Mean = np.zeros_like(x) #Average Matrix
for j in range(k):
    for i in range(nb_groups):
        Mean[int(nn_cumul[i]):int(nn_cumul[i+1]),j] = np.mean(x[int(nn_cumul[i]):int(nn_cumul[i+1]),j])#Average of each variable in each group
    

#set method of classifications
debut = time.time()             
models = [MLPClassifier(alpha=1, random_state=123, max_iter=1000, activation = 'logistic'),
          LogisticRegression(solver='lbfgs', random_state=123, multi_class='multinomial'),
          RDF(max_depth=5, n_estimators=10, max_features=2),
          LinearDiscriminantAnalysis(),
          SVC(kernel='rbf', C=1.0),
          SVC(gamma=2, C=1),
          KNeighborsClassifier(n_neighbors=3),
          GaussianNB(),
          RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), fit_intercept =  False),
          DecisionTreeClassifier(random_state=0),
          AdaBoostClassifier(),
          QuadraticDiscriminantAnalysis()
          ] #All the classifier
    
max_alpha = 5
iteration = 50
A = np.zeros((iteration,3))             
l = -1
#Browse the alpha between (0.1, 5.1) 
for h in np.arange(0.1, max_alpha+0.1, 0.1):
    l = l + 1
    alpha = h
    model = Lda_alpha(alpha)
    #Projection with the intergroup variance matrices
    Z = preprocessing.scale(x)
    V_inter = (1/n)*(Mean.T @ Mean)
    _, P = linalg.eig(V_inter)
    F = np.real(Z @ P) #Projection
    #Applicate the phi_alpha function
    model = Lda_alpha(alpha) 
    F_alpha = model.phi_alpha_matrix(F, alpha)
                
    #Number of train test
    k_fold = 50 
    #Results of the 12 classifier
    ensemble_nonlin = np.zeros((12,k_fold)) 
    for j in range(k_fold):
        i=-1
        #Split the database by training the classifier on 75% of the database
        X_train, X_test, y_train, y_test =train_test_split(F_alpha, y, train_size=0.75)
        for clf in models:
            i = i+1
            ensemble_nonlin[i,0] = i
            #Training of the classifier
            clf.fit(X_train,y_train)
            #Prediction of the classifier
            y_pred = clf.predict(X_test) 
            #Mesure the precision of the test (F-measure)
            ensemble_nonlin[i,j] = f1_score(y_test, y_pred, average='micro')
    #Put all the alpha on the first column
    A[l,0] = l/10+0.1
    #Put the average of the F-measure on the second column
    A[l,1] = np.mean(ensemble_nonlin[:,1::])
    #Put the standard deviation on the thrid column
    A[l,2] = np.std(ensemble_nonlin[:,1::])

#Print F-measure
print(ensemble_nonlin) 
#Print the time of the program
print(time.time() - debut) 
print("moyenne / standard deviation")
print(A)

#Create a graphic to see the evolution with the different alpha
plt.bar(ensemble_nonlin[:,0], ensemble_nonlin[i,1])
plt.xticks(np.arange(k)+1)
plt.show()