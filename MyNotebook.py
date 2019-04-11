
# coding: utf-8

# <a href="https://www.bigdatauniversity.com"><img src="https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png" width="400" align="center"></a>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>

# In this notebook we try to practice all the classification algorithms that we learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Lets first load required libraries:

# In[3]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().magic(u'matplotlib inline')


# ### About dataset

# This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |

# Lets download the dataset

# In[4]:


get_ipython().system(u'wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# ### Load Data From CSV File  

# In[5]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[6]:


df.shape


# ### Convert to date time object 

# In[7]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 
# 

# Let’s see how many of each class is in our data set 

# In[8]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection 
# 

# Lets plot some columns to underestand data better:

# In[9]:


# notice: installing seaborn might takes a few minutes
get_ipython().system(u'conda install -c anaconda seaborn -y')


# In[10]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[11]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[ ]:





# # Pre-processing:  Feature selection/extraction

# ### Lets look at the day of the week people get the loan 

# In[12]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 

# In[13]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values

# Lets look at gender:

# In[14]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Lets convert male to 0 and female to 1:
# 

# In[15]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding  
# #### How about education?

# In[16]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Feature befor One Hot Encoding

# In[17]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame 

# In[18]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature selection

# Lets defind feature sets, X:

# In[19]:


X = Feature
X[0:5]


# What are our lables?

# In[25]:


df['loan_status'].replace(to_replace=['PAIDOFF','COLLECTION'], value=[1,0],inplace=True)
y = df['loan_status'].values
y[0:5]


# In[26]:


a = df['loan_status'].value_counts()
a


# In[27]:


class_wt = {0: np.round(a[1]/np.sum(a), 2), 1: np.round(a[0]/np.sum(a), 2)}
class_wt


# ## Normalize Data 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# In[21]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[23]:


# for AUC vs hyperparamter plot
def auc_hype_plot(k, train_auc, cv_auc, hyper = 'k'):
    plt.figure(figsize = (10,6))
    plt.plot(k, train_auc, 'bo-', label = 'Train AUC')
    plt.plot(k, cv_auc, 'go-', label = 'cv AUC')
    plt.title('AUC of train and cv v/s '+str(hyper)+'-hypreparameter', fontsize = 17)
    plt.legend(fontsize = 15)
    plt.grid('on')
    plt.xlabel('hyperparameter-'+str(hyper), fontsize = 15)
    plt.ylabel('AUC', fontsize = 15)
    plt.show()


# # Classification 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 
# 
# 
# __ Notice:__ 
# - You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# - You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# - You should include the code of the algorithm in the following cells.

# # K Nearest Neighbor(KNN)
# Notice: You should find the best k to build the model with the best accuracy.  
# **warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.

# In[1]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# In[28]:


K = [1,5,10,25,50,75,100,150,200,220]
parameters = dict({'n_neighbors': K})

knn = KNeighborsClassifier(algorithm= 'auto', n_jobs=-1)
clf = GridSearchCV(knn, parameters, scoring = 'roc_auc', cv = 10)
clf.fit(X, y)

train_auc= clf.cv_results_['mean_train_score']
cv_auc = clf.cv_results_['mean_test_score']

auc_hype_plot(K, train_auc, cv_auc)


# Best value of k is 10 becasue cv is giving best AUC at k = 10

# In[29]:


knn_clf = KNeighborsClassifier(n_neighbors = 50, algorithm= 'auto', n_jobs=-1)
knn_clf.fit(X,y)


# # Decision Tree

# In[31]:


from sklearn.tree import DecisionTreeClassifier

depth = [1, 2, 3, 4, 5, 7, 10, 20, 30, 50, 100]
parameters = dict({'max_depth': depth})

DT = DecisionTreeClassifier(class_weight = class_wt)
clf = GridSearchCV(DT, parameters, scoring = 'roc_auc', cv = 10)
clf.fit(X, y)

train_auc= clf.cv_results_['mean_train_score']
cv_auc = clf.cv_results_['mean_test_score']

auc_hype_plot(depth, train_auc, cv_auc, hyper = 'depth')


# #best depth is 2

# In[32]:


DT_clf = DecisionTreeClassifier(max_depth = 2, class_weight = class_wt)
DT_clf.fit(X,y)


# # Support Vector Machine

# In[33]:


from sklearn.linear_model import SGDClassifier

c = np.array([10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 1, 10, 100, 1000, 10000])
parameters = dict({ 'alpha': c})
                  
sgd = SGDClassifier(loss = 'hinge', penalty = 'l2', class_weight = class_wt)
clf = GridSearchCV(sgd, parameters, scoring = 'roc_auc', cv = 10)
clf.fit(X, y)

train_auc= clf.cv_results_['mean_train_score']
cv_auc = clf.cv_results_['mean_test_score']

auc_hype_plot(np.log10(c), train_auc, cv_auc, hyper = 'log10(alpha)')


# Best value of alpha is 10**0

# In[34]:


svm_clf = SGDClassifier(loss = 'hinge', penalty = 'l2', alpha = 10**0, class_weight = class_wt)
svm_clf.fit(X, y)


# # Logistic Regression

# In[35]:


from sklearn.linear_model import LogisticRegression

lamda = np.array([10**-4, 10**-3, 10**-2, 10**-1, 1, 10, 100, 1000, 10000])
c = 1/lamda[::-1] #[::-1] is used to return the array in reverse order
parameters = dict({ 'C': c})

LR = LogisticRegression(penalty = 'l2', class_weight= class_wt) #L2 norm
clf = GridSearchCV(LR, parameters, scoring = 'roc_auc', cv = 5)
clf.fit(X, y)

train_auc= clf.cv_results_['mean_train_score']
cv_auc = clf.cv_results_['mean_test_score']

auc_hype_plot(np.log10(c), train_auc, cv_auc, hyper = 'log10(C)')


# best value of hyperparameter is C = 10**-2 i.e. lambda = 1/C = 100

# In[36]:


LR_clf = LogisticRegression(penalty = 'l2', C = 10**-2, class_weight= class_wt) #L2 norm
LR_clf.fit(X,y)


# # Model Evaluation using Test set

# In[37]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:

# In[38]:


get_ipython().system(u'wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation 

# In[39]:


test_df = pd.read_csv('loan_test.csv')
print(test_df.shape)
test_df.head()


# In[40]:


test_df['due_date'] = pd.to_datetime(test_df['due_date'])#to change into date format

test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])#to change into date format

test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek #converts The day of the week with Monday=0, Sunday=6

test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True) #gender categorical variable into numbers

test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0) #converting day of week into weekend
test_df.head()


# In[41]:


test_Feature = test_df[['Principal','terms','age','Gender','weekend']]
test_Feature = pd.concat([test_Feature, pd.get_dummies(test_df['education'])], axis=1)
test_Feature.drop(['Master or Above'], axis = 1,inplace = True) # to drop dummy variable, we can drop any columns

print(test_Feature.shape)
test_Feature.head()


# In[42]:


x_test = test_Feature
x_test= preprocessing.StandardScaler().fit(x_test).transform(x_test)
x_test[0:5]


# In[43]:


test_df['loan_status'].replace(to_replace=['PAIDOFF','COLLECTION'], value=[1,0],inplace=True)
y_test = test_df['loan_status'].values
print(y_test.shape)
y_test[0:5]


# For k-NN

# In[44]:


y_pred = knn_clf.predict(x_test)
jacc_knn = jaccard_similarity_score(y_test, y_pred)
f1_knn = f1_score(y_test, y_pred)
log_loss_knn = log_loss(y_test, y_pred)
print('For k-NN:\njaccard similarity is {}\nf_score is {}\nlog_loss is {}'.format(jacc_knn, f1_knn, log_loss_knn))


# For Decision Tree
# 

# In[45]:


y_pred = DT_clf.predict(x_test)
jacc_DT = jaccard_similarity_score(y_test, y_pred)
f1_DT = f1_score(y_test, y_pred)
log_loss_DT = log_loss(y_test, y_pred)
print('For k-NN:\njaccard similarity is {}\nf_score is {}\nlog_loss is {}'.format(jacc_DT, f1_DT, log_loss_DT))


# For SVM

# In[46]:


y_pred = svm_clf.predict(x_test)
jacc_svm = jaccard_similarity_score(y_test, y_pred)
f1_svm = f1_score(y_test, y_pred)
log_loss_svm = log_loss(y_test, y_pred)
print('For k-NN:\njaccard similarity is {}\nf_score is {}\nlog_loss is {}'.format(jacc_svm, f1_svm, log_loss_svm))


# For Logistic regression

# In[47]:


y_pred = LR_clf.predict(x_test)
jacc_LR = jaccard_similarity_score(y_test, y_pred)
f1_LR = f1_score(y_test, y_pred)
log_loss_LR = log_loss(y_test, y_pred)
print('For k-NN:\njaccard similarity is {}\nf_score is {}\nlog_loss is {}'.format(jacc_LR, f1_LR, log_loss_LR))


# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:

# | Algorithm          | Jaccard | F1-score | LogLoss |
# |--------------------|---------|----------|---------|
# | KNN                | ?       | ?        | NA      |
# | Decision Tree      | ?       | ?        | NA      |
# | SVM                | ?       | ?        | NA      |
# | LogisticRegression | ?       | ?        | ?       |

# In[48]:


get_ipython().system(u'conda install -c conda-forge prettytable')


# In[49]:


from prettytable import PrettyTable
x = PrettyTable()

x.field_names = ['Algorithms', 'Jaccard', 'F1-score', 'log-loss']
x.add_row(['k-NN', jacc_knn, f1_knn, log_loss_knn])
x.add_row(['Decision Tree', jacc_DT, f1_DT, log_loss_DT])
x.add_row(['SVM', jacc_svm, f1_svm, log_loss_svm])
x.add_row(['Logistic Regression', jacc_LR, f1_LR, log_loss_LR])

print(x)


# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# <p>Copyright &copy; 2018 <a href="https://cocl.us/DX0108EN_CC">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/">MIT License</a>.</p>
