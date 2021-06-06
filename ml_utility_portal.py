'''
Note:  This module was extracted from my pipeline.(put it to the data folder).
=============================================================================================
|        A general utility function to select the best model and tune parameters            |
|        for various classification and regression ML algorithms                            |
|                   By Huanwang Henry Yang  (2016-08-15)                                    |
=============================================================================================

How to use it?
1. import this module to your python code, for example: 

tune_path='C:/Users/hyang978/data-science/Pyth/ML-tune/'
sys.path.append(tune_path)
import ml_tune as tune

2. Tune a perticular model : 
2a. For classification,  use tune.tune_classifier(X_train, y_train, X_test, y_test)
2b. For regression,      use tune.tune_regressor(X_train, y_train, X_test, y_test)

3. Tune all the models : 
3a. For classification,   use tune.tune_classifier_all(X_train, y_train, X_test, y_test)
3b. For regression,       use tune.tune_regressor_all(X_train, y_train, X_test, y_test)

'''
#===================================================================================
#
# -------------below for regression-------------
from  sklearn.ensemble import RandomForestRegressor
from  sklearn.ensemble import ExtraTreesRegressor
from  sklearn.ensemble.weight_boosting import AdaBoostRegressor
from  sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from  sklearn.neighbors.regression import KNeighborsRegressor
from  sklearn.neighbors.regression import RadiusNeighborsRegressor #
from  sklearn.linear_model.base import LinearRegression
from  sklearn.linear_model import SGDRegressor
from  sklearn.linear_model import Ridge
from  sklearn.linear_model import Lasso
from  sklearn.linear_model import ElasticNet
from  sklearn.linear_model import BayesianRidge
from  sklearn.tree.tree import DecisionTreeRegressor
from  sklearn.neural_network import MLPRegressor
from  sklearn.svm.classes import SVR
from  sklearn.svm.classes import LinearSVR
from  xgboost import XGBRegressor

# -------------below for classification-----------------
from  sklearn.ensemble import RandomForestClassifier
from  sklearn.ensemble.weight_boosting import AdaBoostClassifier
from  sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from  sklearn.ensemble import ExtraTreesClassifier
from  xgboost import XGBClassifier
from  sklearn.tree.tree import DecisionTreeClassifier
from  sklearn.neighbors.classification import KNeighborsClassifier #often used
from  sklearn.neighbors.classification import RadiusNeighborsClassifier  # if data not uniformly sampled
from  sklearn.linear_model import LogisticRegression
from  sklearn.linear_model import Perceptron
from  sklearn.linear_model import SGDClassifier
from  sklearn.svm.classes import LinearSVC
from  sklearn.svm.classes import SVC
from  sklearn.naive_bayes import GaussianNB
from  sklearn.naive_bayes import BernoulliNB  #for binary/boolean features
from  sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from  sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from  sklearn.gaussian_process import GaussianProcessClassifier
from  sklearn.neural_network import MLPClassifier
#from imblearn.ensemble import BalancedRandomForestClassifier  #for imbalance data

# library below is used for generating the results and validations ...
from sklearn.metrics import roc_auc_score   #for classifier
from sklearn.metrics import classification_report   #for classifier
from sklearn.metrics import confusion_matrix   #for classifier
from sklearn.metrics import accuracy_score   #for classifier
from sklearn.metrics import mean_squared_error, r2_score  #for regressor

from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import time

from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import  roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#===================================================================================
####----------------------------------------
def tune_classifier_all(X_train, y_train, X_test, y_test,tune):
    ''' Tuning ML algorithms for categorical data  (classification)
    X_* , y_* : the training and test data set. return result as list.
    tune: a key to indicate tune (=1) or not tune (=0) the hyper-parameters
    '''

    classifier=[
     'RandomForestClassifier'
 #   ,'BalancedRandomForestClassifier'
    ,'AdaBoostClassifier'
    ,'GradientBoostingClassifier'
    ,'ExtraTreesClassifier'
    ,'XGBClassifier'
    ,'DecisionTreeClassifier'
    ,'KNeighborsClassifier'
    ,'LogisticRegression'
    ,'GaussianNB'
    ,'BernoulliNB'
    ,'LinearDiscriminantAnalysis'
    ,'MLPClassifier'
    ,'LinearSVC'   
#    ,'QuadraticDiscriminantAnalysis'
#    ,'SVC'   #takes too much memory for large files   
#    ,'Perceptron'
#    ,'GaussianProcessClassifier'  #take all the memory for large data set! remove it
#    ,'RadiusNeighborsClassifier'  #
     ]
    
#    classifier=['KNeighborsClassifier','ExtraTreesClassifier']
    
    all_result=[]
    for model in classifier: 
        result=tune_classifier(model, X_train, y_train, X_test, y_test,tune)
        all_result.append(result)
        
    best=sorted(all_result, key=lambda x: x[2], reverse=True)
    columns=['model_inp', 'train_score','test_score','cpu','para','model']
    df_results = pd.DataFrame(best, columns=columns)
  
    print('==================================================================')
    print('\nThe table for each model\n', df_results.iloc[:,0:4], '\n')
    print('\nThe best model=', df_results.iloc[0,5])

#--------------------------------------------------
def tune_regressor_all(X_train, y_train, X_test, y_test,tune):
    ''' Tuning ML algorithms for numerical data 
    X_* , y_* : the training and test data set. return result as list.
    tune: a key to indicate tune (=1) or not tune (=0) the hyper-parameters

    '''

    regressor=[
    'RandomForestRegressor'
   ,'AdaBoostRegressor'
   ,'GradientBoostingRegressor'
   ,'ExtraTreesRegressor'
   ,'XGBRegressor'
   ,'DecisionTreeRegressor'
   ,'KNeighborsRegressor'
   ,'MLPRegressor'
   ,'LinearRegression'
   ,'Ridge'
   ,'Lasso'
   ,'ElasticNet'
   ,'BayesianRidge'
#   ,'SVR'
#   ,'LinearSVR'
#   ,'SGDRegressor'

    ]
    
    all_result=[]
    for model in regressor: 
        result=tune_regressor(model, X_train, y_train, X_test, y_test,tune)
        all_result.append(result)
        
    best=sorted(all_result, key=lambda x: x[2], reverse=True)
    columns=['model_inp', 'train_score','test_score','cpu','para','model']
    df_results = pd.DataFrame(best, columns=columns)
  
    print('==================================================================')
    print('\nThe table for each model\n', df_results.iloc[:,0:4], '\n')
    print('\nThe best model:\n model=', df_results.iloc[0,5])

####-------------------------------------------------
def tune_regressor(model, X_train, y_train, X_test, y_test, tune=1):
    '''Using *_train, tuning various popular ML algorithm for regression
    X_* , y_* : the training and test data set. return result as list.
    tune: a key to indicate tune (=1) or not tune (=0) the hyper-parameters
    '''
    
    time_start = time.clock()
    print('----------------------------------------------------------------------------\n')   
    print( '\nTuning hyperparameters for ', model)
    
    if model=='RandomForestRegressor':
       hyper_para=dict(criterion=['mse', 'mae'], max_depth=[8,6,None], n_estimators=[150],
                  max_features=['auto','sqrt'])

       mod = RandomForestRegressor()

    elif model=='ExtraTreesRegressor':
       hyper_para=dict(criterion=[ 'mse','mae'], max_depth=[8,6, None], n_estimators=[150],
                  max_features=['auto','sqrt'])

       mod = ExtraTreesRegressor()

    elif model=='AdaBoostRegressor':
       hyper_para=dict(n_estimators=[150],learning_rate=[1.0,0.7], loss=['linear', 'square', 'exponential'])

       mod = AdaBoostRegressor()

    elif model=='GradientBoostingRegressor':
       hyper_para=dict(n_estimators=[150],learning_rate=[0.1,0.05], loss=['ls', 'lad', 'huber'],
                  max_features= [ 'auto','sqrt'])

       mod = GradientBoostingRegressor()

    elif model=='DecisionTreeRegressor':
       hyper_para=dict(splitter=['best', 'random'], max_depth=[5,4,3,2,None], max_features=
                  ['auto','sqrt'])
       mod = DecisionTreeRegressor()

    elif model=='KNeighborsRegressor':
       hyper_para=dict(n_neighbors=list(range(1, 30)), weights=['uniform', 'distance'])
       mod = KNeighborsRegressor()  #by default
   
    elif model=='MLPRegressor':
       hyper_para=dict(solver=['lbfgs', 'sgd', 'adam'])
       mod = MLPRegressor()

    elif model=='SGDRegressor':
       hyper_para=dict(loss=['squared_loss', 'huber'])
       mod = SGDRegressor()

    elif model=='LinearRegression':
       hyper_para=dict()
       mod = LinearRegression()  #by default

    elif model=='Ridge':
       hyper_para=dict(solver=['auto'], alpha=[1.0, 2.0])
       mod = Ridge()

    elif model=='Lasso':
       hyper_para=dict(alpha=[1.0])
       mod = Lasso()
    
    elif model=='ElasticNet':
       hyper_para=dict( l1_ratio=[0, 0.3, 0.5, 0.7, 1.0])
       mod = ElasticNet()    

    elif model=='BayesianRidge':
       hyper_para=dict()
       mod = BayesianRidge()    

    elif model=='SVR':
       hyper_para=dict()
       mod = SVR()
    
    elif model=='LinearSVR':
       hyper_para=dict()
       mod = LinearSVR()
 
    elif model=='XGBRegressor':
       hyper_para=dict()
       mod = XGBRegressor()
        
    if tune==0 : hyper_para=dict()
    grid = do_GridSearchCV(mod, X_train, y_train, X_test, y_test, hyper_para, 'reg')

#    plot_learning_curve(grid, "{}".format(model), X_train, y_train, ylim=(0.75,1.0), cv=5)   
        
    time_end = time.clock()
    time_dif = time_end - time_start
    
    best_train_score = grid.score(X_train, y_train)
    best_test_score = grid.score(X_test, y_test)
        
    print('\nbest_train_score={tr:.3f}: best_test_score={tt:.3f} : CPU time= {t:.2f} s'.format(tr=grid.best_score_,tt= best_test_score, t=time_dif))
    print('best_params=', grid.best_params_)
    print('model=', grid.best_estimator_)
    
    result=[model, best_train_score, best_test_score, time_dif, grid.best_params_, grid.best_estimator_]
    return result
 
####------------------------------------------------
def tune_classifier(model, X_train, y_train, X_test, y_test, tune=1):
    ''' Using *_train, tuning various popular ML algorithm for classification 
    X_* , y_* : the training and test data set. return result as list
    tune: a key to indicate tune (=1) or not tune (=0) the hyper-parameters    
    '''
    print('\n------------------------------------------------------------------')           
    print( '\nTuning hyperparameters for ', model)
    
    time_start = time.clock()
    if model=='RandomForestClassifier':
       h_para=dict(criterion=['gini', 'entropy'], max_depth=[6,4,2, None], n_estimators=[200],
              max_features=[ 'auto',None])
       mod=RandomForestClassifier()
       
    elif model=='BalancedRandomForestClassifier':
       h_para=dict(criterion=['gini', 'entropy'], max_depth=[6,4,2, None], n_estimators=[200],
              max_features=[ 'auto',None])
       mod=BalancedRandomForestClassifier()

    elif model=='ExtraTreesClassifier':
       h_para=dict(criterion=['gini', 'entropy'], max_depth=[6,4,None], n_estimators=[200],
                  max_features=[ 'auto',None])
       mod=ExtraTreesClassifier()

    elif model=='XGBClassifier':
       h_para=dict(max_depth=[2,5,7, 9],subsample=[1],n_estimators=[200],
                  colsample_bytree=[ 1])
       mod = XGBClassifier( )
        
    elif model=='AdaBoostClassifier':
       h_para=dict(n_estimators=[200], algorithm=['SAMME', 'SAMME.R'])
       mod = AdaBoostClassifier()

    elif model=='GradientBoostingClassifier':
       h_para=dict(n_estimators=[200],learning_rate=[1.0], loss=['deviance'],
                  max_features= ['auto',None], max_depth=[5,4,3])
       mod = GradientBoostingClassifier()

    elif model=='DecisionTreeClassifier':
       h_para=dict(splitter=['best', 'random'], max_depth=[5,4,3,None], max_features=
                  ['auto', None])
       mod = DecisionTreeClassifier()

    elif model=='KNeighborsClassifier':
       h_para = dict(n_neighbors=[3,5,7,9,11,13,15], weights=['uniform', 'distance'])
       mod=KNeighborsClassifier()  #by default

    elif model=='RadiusNeighborsClassifier':
       h_para = dict(radius=[0.5, 1, 2, 5], weights=[ 'distance'])
       mod=RadiusNeighborsClassifier()  #by default

    elif model=='LogisticRegression':
       h_para=dict(penalty=['l2'], class_weight=[None, 'balanced'], 
                  solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
       mod = LogisticRegression()

    elif model=='LinearDiscriminantAnalysis':
       h_para=dict(solver=['svd', 'lsqr', 'eigen'])
       mod = LinearDiscriminantAnalysis()

    elif model=='SVC':
       h_para=dict(kernel=['linear', 'poly', 'rbf'], decision_function_shape=[ 'ovo', 'ovr'],
                      Cs = [0.001, 0.01, 0.1, 1, 10],gammas = [0.001, 0.01, 0.1, 1],
                        param_grid = {'C': Cs, 'gamma' : gammas})
       mod = SVC()

    elif model=='LinearSVC':
       h_para=dict(penalty=[ 'l2'], loss=['hinge', 'squared_hinge'])
       mod = LinearSVC()

    elif model=='BernoulliNB':
       h_para=dict()
       mod = BernoulliNB()

    elif model=='GaussianNB':
       h_para=dict()
       mod = GaussianNB()

    elif model=='Perceptron':
       h_para=dict()
       mod = Perceptron()

    elif model=='SGDClassifier':
       h_para=dict(loss=['hinge', 'log', 'modified_huber'], penalty=['l2', 'l1', 'elasticnet'])
       mod = SGDClassifier()

    elif model=='QuadraticDiscriminantAnalysis':
       h_para=dict()
       mod = QuadraticDiscriminantAnalysis()
 
    elif model=='GaussianProcessClassifier':
       h_para=dict()
       mod = GaussianProcessClassifier()

    elif model=='MLPClassifier':
       h_para=dict(solver=['lbfgs', 'sgd', 'adam'])
       mod = MLPClassifier()
        
    if tune==0 : h_para=dict()
    grid = do_GridSearchCV(mod, X_train, y_train, X_test, y_test, h_para,'class')

#    plot_learning_curve(grid, "{}".format(model), X_train, y_train, ylim=(0.75,1.0), cv=10)   
        
    time_end = time.clock()
    time_dif = time_end - time_start
    
    best_train_score = grid.score(X_train, y_train)
    best_test_score = grid.score(X_test, y_test)
        
    print('\nbest_train_score={tr:.3f}: best_test_score={tt:.3f} : CPU time= {t:.2f} s'.format(tr=grid.best_score_,tt= best_test_score, t=time_dif))
    print('best_params=', grid.best_params_)
    print('model=', grid.best_estimator_)
    
    result=[model, best_train_score, best_test_score, time_dif, grid.best_params_, grid.best_estimator_]
    return result

####-------------------------------------------------
def do_GridSearchCV(model, X_train, y_train, X_test, y_test, param_grid, type):
    '''model: the given model (regressor or clasifier)
    X_: a data frame containing all the features (excep target) (train | test)
    y_: the target (or class) (train | test)
    param_grid: a dictionary containing each hyper-parameter
    type: reg (for regressor) or class (classification)
    '''
    
    run_param=param_grid 
    if len(param_grid)==0 : run_param=dict()  #include default
 
    print('input_params=', run_param)
    if type == 'reg' :  #regression
        grid = GridSearchCV(model, run_param, cv=4,  n_jobs=-1) #use all cpu
    elif type =='class': #classification
        grid = GridSearchCV(model, run_param, cv=4, scoring='accuracy', n_jobs=-1) 
    else:
        print( 'Error: please indicate "reg" or "class" for the GridSearchCV.')
        return 
    estimator=grid.fit(X_train, y_train)

    return estimator

####-------------------------------------------------
 
def plot_dist_boxplot_class(df, target):
    '''df: the data frame;  target: the column name for the target (class)
    '''
    
    import matplotlib.gridspec as gridspec
    
    data = df.select_dtypes(include=[np.number])  #only for numerical data

    data['class_plot']='class_' + data[target].astype(str)
    print('The class values=', data[target].unique())
    for feature in data.columns:
        if feature == target or feature == 'class_plot' : continue
        print('ploting ', feature)
        gs1 = gridspec.GridSpec(3,1)
        ax1 = plt.subplot(gs1[:-1])
        ax2 = plt.subplot(gs1[-1])
        gs1.update(right=1.00)
        sns.boxplot(x=feature,y='class_plot',data=data,ax=ax2)
        for i in data[target].unique():
            sns.kdeplot(data[feature][data[target]==i],ax=ax1,label=str(i))

        ax2.yaxis.label.set_visible(False)
        ax1.xaxis.set_visible(False)
        plt.show()
####-------------------------------------------------
def impute_by_knn(df, target=False, n_neighbor=5):
  '''impute missing data by kNN
  df is either a array or a df.  (prefered)
  '''

  from sklearn.impute import KNNImputer

  imputer = KNNImputer(n_neighbors=n_neighbor, weights="uniform")
  if not target :   #warning: may be biased
    print('Imputing data based on all columns of DF')
    d=imputer.fit_transform(df)   #as numpy  
    return pd.DataFrame(d, columns=df.columns) #back to DF
  print('Imputing data without target')
  X=df.drop(target, axis=1)
  y=df[target]
  #d=imputer.fit_transform(np.array(df))
  d=imputer.fit_transform(X)   #as numpy
  
  df1=pd.DataFrame(d, columns=X.columns) #back to DF
  dfn=pd.concat([df1, y], axis=1)

  return dfn
####-------------------------------------------------
def find_outliers(df, scale=10):
  '''list the number of outliers using boxplot, but use large scale
  '''
  df_num=df.select_dtypes(include='number') #only for numerical

  for col_name in df_num.columns:  
    q1 = df_num[col_name].quantile(0.25)
    q3 = df_num[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-scale*iqr
    fence_high = q3+scale*iqr

    n=0
    for val in df[col_name]:
      if val>fence_high or val<fence_low : n=n+1
    if n >0 :
      print('{}:   value beyond ({:.2f}  {:.2f})   outliers ={}: ' \
      .format(col_name, fence_low, fence_high, n))

####-------------------------------------------------
def handle_outliers(df_in, action='drop', scale=10):
  '''A function to eigher drop or wisorize the outlier
  '''

  df=df_in.select_dtypes(include='number') #must be num
  for x in df.columns:
    iqr=df[x].quantile(.75) - df[x].quantile(.25)
    fence_low  = df[x].quantile(.25) - scale*iqr
    fence_high = df[x].quantile(.75) + scale*iqr
    if action=='drop':  #drop
      df.drop(df[df[x]>fence_high].index, inplace=True)
      df.drop(df[df[x]<fence_low].index, inplace=True)
    elif action=='keep':  #use clip 
      df[x] = df[x].clip(fence_low,fence_high)

  return df

####-------------------------------------------------
def plot_feature_importance(X, y, model=RandomForestClassifier(),nfeature=30):
    '''plot important features
    X: the training data;  y the target for training data
    '''

    model=RandomForestClassifier()
    model.fit(X,y)
#    print(model.feature_importances_)  

    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat=feat_importances.nlargest(nfeature).sort_values(ascending=True)
    feat_importances.nlargest(nfeature).sort_values(ascending=True).plot(kind='barh')
    plt.show() 
    #feature=feat[feat>0.03].index
    return feat

####-------------------------------------------------
#--------------------------------------------------------

def get_feature_importance(model, feature_name):
    df = pd.DataFrame({'feature': list(feature_name),
                       'importance': model.feature_importances_}).\
                        sort_values('importance', ascending = False)
    print(df.head(len(feature_name)))
#--------------------------------------------------------
def transf(select, X_train, y_train, X_test, y_test):
    '''
    '''
    select.fit(X_train, y_train)
    X_train_s=select.transform(X_train)
    X_test_s=select.transform(X_test)
    print( 'The shape of X_train = ', X_train.shape)
    print( 'The shape of X_train_select = ', X_train_s.shape)

    return X_train_s, X_test_s
    
#--------------------------------------------------------
def feature_selection(X_train, X_test, y_train, y_test, ftype, nfeature):
    '''select the best feature to reduce noise, overfit
        X & y are the train, test for feature and target
        ftype: feature type ("model", "RFE");  
        nfeature: number of features (like 20)

Recursive Feature Elimination (RFE) is based on the idea to repeatedly construct 
a model and choose either the best or worst performing feature, setting the feature 
aside and then repeating the process with the rest of the features. This process 
is applied until all features in the dataset are exhausted. The goal of RFE is to 
select features by recursively considering smaller and smaller sets of features.

It enables the machine learning algorithm to train faster.
It reduces the complexity of a model and makes it easier to interpret.
It improves the accuracy of a model if the right subset is chosen.
It reduces overfitting.

refer to https://scikit-learn.org/stable/modules/feature_selection.html

    '''

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import RFE
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestRegressor
    from  xgboost import XGBClassifier
    from sklearn.feature_selection import VarianceThreshold

#Univariate feature selection works by selecting the best features 
# based on univariate statistical tests
    print("Feature selection:  using ftype=", ftype, "nfeature=", nfeature)
    if ftype=='VAR': 
        p=nfeature  #p =0.01 , remove the column having 99% same value 
        select = VarianceThreshold(threshold=p)
        
    elif ftype=='CHI2': 
        select = SelectKBest(score_func=chi2, k=nfeature)

    elif ftype=='RFE_RF': #  Recursive Feature Elimination (RFE)
        model = RandomForestRegressor(n_jobs=-1)
        select = RFE(model, nfeature)      
    elif ftype=='RFE_XGB': #  Recursive Feature Elimination (RFE)
        model = XGBClassifier(n_jobs=-1)
        select = RFE(model, nfeature)

    elif ftype=='model_RF':  #model RF based method, threshold=None
        model=RandomForestRegressor(n_jobs=-1)  #use all cores
        if nfeature==0:
            select=SelectFromModel(model, threshold=None)
        else:
            select=SelectFromModel(model, threshold=-np.inf, max_features= nfeature)
    elif ftype=='model_XGB':  #model XGB based method
        model=XGBClassifier(n_jobs=-1)
        if nfeature==0:
            select=SelectFromModel(model, threshold=None)
        else:
            select=SelectFromModel(model, threshold=-np.inf, max_features= nfeature)

    X_train_s, X_test_s = transf(select, X_train, y_train, X_test, y_test)

#the transf gives a array. I want it to be a dataframe (keep selected features)
    feature_idx = select.get_support()
    feature_name = X_train.columns[feature_idx]
    X_train_s=pd.DataFrame(X_train_s, columns=feature_name)  #mkae a df
    X_test_s=pd.DataFrame(X_test_s, columns=feature_name)  #mkae a df
    model.fit(X_train_s, y_train)
    get_feature_importance(model, feature_name)
 #   model.fit(X_train, y_train)
 #   get_feature_importance(model, list(X_train.columns))
    
    return X_train_s, X_test_s    
#--------------------------------------------------------  
#Classification results
def write_result_class(X_test, y_test, y_pred, model):
    '''X_test: test set containing features only
       y_test: test set containing target only
       y_pred: the predicted values corresponding to the y_test
       model:  the model used to train the data (X_train)
    '''


    #for i in range(len(y_pred)): print( 'predicted, target=', y_pred[i],y_test.values[i])

    print( '\nConfusion_matrix=\n', confusion_matrix(y_test, y_pred))
    print( 'Classification_report=\n', classification_report(y_test, y_pred))

 #   if len(y_test.unique())>2: return #below for binary class

    model_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    print( 'Classification accuracy=', model.score(X_test, y_test))
    print( 'Classification AUC_ROC= ', model_roc_auc)
 
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='AUC_ROC (area = %0.2f)' % model_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('AUC_ROC')
    plt.show()

    
