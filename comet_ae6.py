# %% In [1]:
from comet_ml import Experiment

#pip install comet_ml

# %% In [2]:
# Create an experiment with your api key

# import comet_ml at the top of your file
from comet_ml import Experiment

# Create an experiment with your api key
experiment = Experiment(
    api_key="MpO4LpB9MQYCpwSMctM8yvQCN",
    project_name="team-ae6",
    workspace="shanipillay",
)

#experiment = Experiment(api_key="4tkzu35HQsdtVizYgCkTOIJ2L",
                        #project_name="classification-ae6-dsft21", workspace="seromo", log_code=True)

# %% In [3]:
# storing and analysis
import numpy as np
import pandas as pd
import re

# visualization
import matplotlib.pyplot as plt
import warnings
import nltk
import string
import seaborn as sns

#import text classification modules
import os
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from nltk.stem.porter import * 
from wordcloud import WordCloud
import spacy
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

# import train/test split module
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# import scoring metrice
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# suppress cell warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

# Standard libraries
import re
import csv
import nltk
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

# Style
import matplotlib.style as style 
sns.set(font_scale=1.5)
style.use('seaborn-pastel')
style.use('seaborn-poster')
from PIL import Image
from wordcloud import WordCloud

# Downloads
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Preprocessing
from collections import Counter
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords, wordnet  
from sklearn.feature_extraction.text import CountVectorizer   
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Building classification models
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Model evaluation
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

# %% In [4]:
#Load the training set and testing set
train = pd.read_csv('train.csv') 
test = pd.read_csv('test_with_no_labels.csv') 
hashtag = pd.read_csv('train.csv') 

# %% In [5]:
#display first 5 entries of the train data
train.head()

# %% In [6]:
#Display the first 5 entries of the test data
test.head()

# %% In [7]:
#Print out the Shape of the training data and the testing data
print('Shape of Train Dataset:',train.shape)
print('Shape of Test Dataset:',test.shape)

# %% In [8]:
#Use the value_counts() method to displace the count of each sentiment in the training dataset
train['sentiment'].value_counts()

# %% In [9]:
#Use the isnull() method to check for null values in training data
#.sum() method evaluates the total of each column of null values
train.isnull().sum()

# %% In [10]:
#Combining both train and test data set before data cleaning as tweets in both the data set is unstructured
data = train.append(test, ignore_index=True) 

# %% In [11]:
def TweetCleaner(tweet):
   
    # Convert to lowercase
    tweet = tweet.lower() 
    
    # Remove mentions or twitter handles   
    tweet = re.sub('@[\w]*','',tweet)  
    
    # Remove url's
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    
    # Remove hashtags
    tweet = re.sub(r'#\w*', '', tweet)    
    
    # Remove numbers
    tweet = re.sub(r'\d+', '', tweet)  
    
    # Remove punctuation
    tweet = re.sub(r"[,.;':@#?!\&/$]+\ *", ' ', tweet)                  
    
    return tweet

# Clean the tweets in the message column
data['message'] = data['message'].apply(TweetCleaner)
data['message'] = data['message'].apply(TweetCleaner)

data.head()

# %% In [12]:
#Use tokenization to the words into a list of tokens 
tokenized_tweet = data['message'].apply(lambda x: x.split()) 
tokenized_tweet.head()

# %% In [13]:
#Use PorterStemmer() to strip suffixes from the words
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer() 

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
tokenized_tweet.head()

# %% In [14]:
# bring the words back together 
for i in range(len(tokenized_tweet)): 
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i]) 

data['message'] = tokenized_tweet 

# %% In [15]:
#Split the dataset back to the training set and the testing set
train = data[:len(train)] 
test = data[len(test):]  

# %% In [16]:
#Check the number of rows and columns in the dataset
train.shape

# %% In [17]:
#The frequency values of the individual sentiments
train['sentiment'].value_counts() 

# %% In [18]:
style.use('seaborn-pastel')

fig, axes = plt.subplots(ncols=2, 
                         nrows=1, 
                         figsize=(20, 10), 
                         dpi=100)

sns.countplot(train['sentiment'], ax=axes[0])

labels=['Pro', 'News', 'Neutral', 'Anti'] 

axes[1].pie(train['sentiment'].value_counts(),
            labels=labels,
            autopct='%1.0f%%',
            shadow=True,
            startangle=90,
            explode = (0.1, 0.1, 0.1, 0.1))

fig.suptitle('Dataset distribution', fontsize=20)
plt.show()

# %% In [19]:
#create a new length value column that contains the lengths of the messages
train['message_length'] = train['message'].apply(len)

#Create a violinplot of the dataset
plt.figure(figsize=(8,5)) #Set the figsize to 8 and 5 respectively
plt.title('Sentiments vs. Length of tweets') #Add the title of the violin plot
sns.violinplot(x='sentiment', y='message_length', data=train,scale='count') #Add the dimentions of the violin plot
plt.ylabel("Length of the tweets") #Y_lable of the plot
plt.xlabel("Sentiment Class") #X_label of the plot

# %% In [20]:
#Use groupby in order to numerically display what the boxplot is trying to show to the user
train['message_length'].groupby(train['sentiment']).describe()

# %% In [21]:
#Create strings for each class
positive_words =' '.join([text for text in data['message'][data['sentiment'] == 1]]) #Words in the positve class
negative_words = ' '.join([text for text in data['message'][data['sentiment'] == -1]]) #Words in negative class
normal_words =' '.join([text for text in data['message'][data['sentiment'] == 0]]) #Words in the neutral class
news_words =' '.join([text for text in data['message'][data['sentiment'] == 2]]) #Words in the news class

# %% In [22]:
#Create a user defined function to display a word cloud for each class
def word_cloud(class_words):
   
    wordcloud = WordCloud(background_color='white',width=800, height=500, random_state=21, max_font_size=110).generate(class_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Most Common words")
    plt.axis('off')
    return plt.show()

# %% In [23]:
#Visualise all words from the positive class
word_cloud(positive_words)

# %% In [24]:
#Visualise all words from the negative class
word_cloud(negative_words)

# %% In [25]:
#Visualise all words from the neutral class
word_cloud(normal_words)

# %% In [26]:
#Visualise all words from the news class
word_cloud(news_words)

# %% In [27]:
#Create a function to collect hashtags
def hashtag_extract(x):
    
    hashtags = [] 
    for i in x:   
        ht = re.findall(r"#(\w+)", i) 
        hashtags.append(ht)

    return hashtags

# %% In [28]:
# extracting hashtags from the news
HT_news = hashtag_extract(hashtag['message'][hashtag['sentiment'] == 2])
# extracting hashtags from positive sentiments
HT_positive = hashtag_extract(hashtag['message'][hashtag['sentiment'] == 1])
# extract hashtags from neutral sentiments
HT_normal = hashtag_extract(hashtag['message'][hashtag['sentiment'] == 0])
# extracting hashtags from negative sentiments
HT_negative = hashtag_extract(hashtag['message'][hashtag['sentiment'] == -1])

# unnesting list of all sentiments
HT_news = sum(HT_news,[])
HT_positive = sum(HT_positive,[])
HT_normal = sum(HT_normal,[])
HT_negative = sum(HT_negative,[])

# %% In [29]:
#Create a function that visualises the barplot distribution of the hashtags
def bar_dist(x):
    
    a = nltk.FreqDist(x) 
    d = pd.DataFrame({'Hashtag': list(a.keys()), 
                  'Count': list(a.values())})  
    d = d.nlargest(columns="Count", n = 10)  
    plt.figure(figsize=(16,5))
    ax = sns.barplot(data=d, x= "Hashtag", y = "Count") 
    ax.set(ylabel = 'Count') 
    return plt.show()

# %% In [30]:
#Display barplot of the News hastags
bar_dist(HT_news)

# %% In [31]:
#Display barplot of the positive hastags
bar_dist(HT_positive)

# %% In [32]:
#Display barplot of the neutral hastags
bar_dist(HT_normal)

# %% In [33]:
#Display barplot of the neutral hastags
bar_dist(HT_negative)

# %% In [34]:
#Splitting features and target variables
X = train['message'] 
y = train['sentiment']    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) 

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# %% In [35]:
# import and call the TFidfVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
tfidf = TfidfVectorizer() 

# %% In [36]:
#import CountVectorizer and call it
from sklearn.feature_extraction.text import CountVectorizer 

cf= CountVectorizer() 

# %% In [37]:
#Import metrics from sklearn
from sklearn import metrics

# %% In [38]:
#Create a barplot for the train dataset classes
News = train['sentiment'].value_counts()[2] 
Pro= train['sentiment'].value_counts()[1]   
Neutral=train['sentiment'].value_counts()[0]
Anti=train['sentiment'].value_counts()[-1]  

sns.barplot(['News ','Pro','Neutral','Anti'],[News,Pro,Neutral,Anti]) 
plt.xlabel('Tweet Classification') 
plt.ylabel('Count of Tweets')      
plt.title('Dataset labels distribution') 
plt.show() 

# %% In [39]:
#Import the resampling module
from sklearn.utils import resample

# %% In [40]:
#Downsample and upsample train dataset

X = train['message']
y = train['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

df_upsampled_train = pd.DataFrame({'message': X_train, 'sentiment': y_train})


df_majority = train[train.sentiment==1] 
df_minority = train[train.sentiment==0] 
df_minority1 = train[train.sentiment==2] 
df_minority2 = train[train.sentiment==-1]

# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    
                                 n_samples=5000,     
                                 random_state=123) 
#Upsampling the least minority class
df_minority_up = resample(df_minority, 
                        replace=True,   
                        n_samples=5000,     
                        random_state=123) 

df_minority_up1 = resample(df_minority1, 
                        replace=True,    
                        n_samples=5000,     
                        random_state=123) 

df_minority_up2 = resample(df_minority2, 
                        replace=True,   
                        n_samples=5000,    
                        random_state=123) 

# Combine minority class with downsampled majority class
df_resampled = pd.concat([df_majority_downsampled,df_minority_up,df_minority_up1, df_minority_up2])
 
# Display new class counts
df_resampled.sentiment.value_counts()

y_train = df_upsampled_train['sentiment']
X_train = df_upsampled_train['message']

# %% In [41]:
#X = df_resampled['message']
#y = df_resampled['sentiment']

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# %% In [42]:
# Random Forest Classifier
rf = Pipeline([('tfidf', TfidfVectorizer()),
               ('clf', RandomForestClassifier(max_depth=5, 
                                              n_estimators=100))])

# Naïve Bayes:
nb = Pipeline([('tfidf', TfidfVectorizer()),
               ('clf', MultinomialNB())])

# K-NN Classifier
knn = Pipeline([('tfidf', TfidfVectorizer()),
                ('clf', KNeighborsClassifier(n_neighbors=5, 
                                             metric='minkowski', 
                                             p=2))])

# Logistic Regression
lr = Pipeline([('tfidf',TfidfVectorizer()),
               ('clf',LogisticRegression(C=1, 
                                         class_weight='balanced', 
                                         max_iter=1000))])
# Linear SVC:
lsvc = Pipeline([('tfidf', TfidfVectorizer()),
                 ('clf', LinearSVC(class_weight='balanced'))])

# %% In [43]:
# Random forest 
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Niave bayes
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# K - nearest neighbors
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Linear regression
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Linear SVC
lsvc.fit(X_train, y_train)
y_pred_lsvc = lsvc.predict(X_test)

# %% In [44]:
# Generate a classification Report for the random forest model
print(metrics.classification_report(y_test, y_pred_rf))
print('accuracy %s' % accuracy_score(y_pred_rf, y_test)) 
print('f1_score %s' % metrics.f1_score(y_test,y_pred_rf,average='weighted'))

# Generate a normalized confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)
cm_norm = cm / cm.sum(axis=1).reshape(-1,1)

# Display the confusion matrix as a heatmap
sns.heatmap(cm_norm, 
            cmap="YlGnBu", 
            xticklabels=rf.classes_, 
            yticklabels=rf.classes_, 
            vmin=0., 
            vmax=1., 
            annot=True, 
            annot_kws={'size':10})

# Adding headings and lables
plt.title('Random forest classification')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# %% In [45]:
# Generate a classification Report for the Naive Bayes model
print(metrics.classification_report(y_test, y_pred_nb))
print('accuracy %s' % accuracy_score(y_pred_nb, y_test)) 
print('f1_score %s' % metrics.f1_score(y_test,y_pred_nb,average='weighted'))

# Generate a normalized confusion matrix
cm = confusion_matrix(y_test, y_pred_nb)
cm_norm = cm / cm.sum(axis=1).reshape(-1,1)

# Display the confusion matrix as a heatmap
sns.heatmap(cm_norm, 
            cmap="YlGnBu", 
            xticklabels=nb.classes_, 
            yticklabels=nb.classes_, 
            vmin=0., 
            vmax=1., 
            annot=True, 
            annot_kws={'size':10})

# Adding headings and lables
plt.title('Naive Bayes classification')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# %% In [46]:
# Generate a classification Report for the K-nearest neighbors model
print(metrics.classification_report(y_test, y_pred_knn))
print('accuracy %s' % accuracy_score(y_pred_knn, y_test)) 
print('f1_score %s' % metrics.f1_score(y_test,y_pred_knn,average='weighted'))

# Generate a normalized confusion matrix
cm = confusion_matrix(y_test, y_pred_knn)
cm_norm = cm / cm.sum(axis=1).reshape(-1,1)

# Display the confusion matrix as a heatmap
sns.heatmap(cm_norm, 
            cmap="YlGnBu", 
            xticklabels=knn.classes_, 
            yticklabels=knn.classes_, 
            vmin=0., 
            vmax=1., 
            annot=True, 
            annot_kws={'size':10})

# Adding headings and lables
plt.title('K - nearest neighbors classification')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# %% In [47]:
# Generate a classification Report for the model
print(metrics.classification_report(y_test, y_pred_lr))
print('accuracy %s' % accuracy_score(y_pred_lr, y_test)) 
print('f1_score %s' % metrics.f1_score(y_test,y_pred_lr,average='weighted'))

cm = confusion_matrix(y_test, y_pred_lr)
cm_norm = cm / cm.sum(axis=1).reshape(-1,1)

sns.heatmap(cm_norm, 
            cmap="YlGnBu", 
            xticklabels=lr.classes_, 
            yticklabels=lr.classes_, 
            vmin=0., 
            vmax=1., 
            annot=True, 
            annot_kws={'size':10})

# Adding headings and lables
plt.title('Logistic regression classification')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# %% In [48]:
# Generate a classification Report for the linear SVC model
print(metrics.classification_report(y_test, y_pred_lsvc))
print('accuracy %s' % accuracy_score(y_pred_lsvc, y_test)) 
print('f1_score %s' % metrics.f1_score(y_test,y_pred_lsvc,average='weighted'))

# Generate a normalized confusion matrix
cm = confusion_matrix(y_test, y_pred_lsvc)
cm_norm = cm / cm.sum(axis=1).reshape(-1,1)

# Display the confusion matrix as a heatmap
sns.heatmap(cm_norm, 
            cmap="YlGnBu", 
            xticklabels=lsvc.classes_, 
            yticklabels=lsvc.classes_, 
            vmin=0., 
            vmax=1., 
            annot=True, 
            annot_kws={'size':10})

# Adding headings and lables
plt.title('Linear SVC classification')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# %% In [49]:
from sklearn.model_selection import GridSearchCV

parameters_svm = {'tfidf__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf__C': (0.001, 0.01, 0.1, 1)}
gs_clf = GridSearchCV(lsvc, parameters_svm, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train)
print("Best cross-validation score: {:.2f}".format(gs_clf.best_score_))
print("Best parameters: ", gs_clf.best_params_)

# %% In [50]:
# This code is intentionally commented out - Code takes >10 minutes to run. 

"""
# Set ranges for the parameters that we want to tune
params = {'clf__C': [0.1, 0.2, 0.3, 0.4, 0.5],
          'tfidf__ngram_range' : [(1,1),(1,2)],
          'clf__max_iter': [1500, 2000, 2500, 3000],
          'tfidf__min_df': [2, 3, 4],
          'tfidf__max_df': [0.8, 0.9]}

# Perform randomized search & extract the optimal parameters
Randomized = RandomizedSearchCV(text_clf_lsvc, param_distributions=params, cv=5, scoring='accuracy', n_iter=5, random_state=42)
Randomized.fit(X_train,y_train)
Randomized.best_estimator_
"""

from sklearn.model_selection import GridSearchCV

# Retrain linear SVC using optimal hyperparameters:
lsvc_op = Pipeline([('tfidf', TfidfVectorizer(max_df=0.8,
                                                    min_df=2,
                                                    ngram_range=(1,2))),
                  ('clf', LinearSVC(C=0.3,
                                    class_weight='balanced',
                                    max_iter=3000))])

# Fit and predict
lsvc_op = GridSearchCV(lsvc, parameters_svm, n_jobs=-1)
lsvc_op.fit(X_train, y_train)
y_pred = lsvc_op.predict(X_test)
print("Best cross-validation score: {:.2f}".format(gs_clf.best_score_))
print("Best parameters: ", gs_clf.best_params_)
print('F1 score improved by',
      round(100*((metrics.accuracy_score(y_pred, y_test) - metrics.accuracy_score(y_pred_lsvc, y_test)) /metrics.accuracy_score(y_pred_lsvc, y_test)),0), 
      '%')
print('Old f1_score %s' % metrics.f1_score(y_test,y_pred_lsvc,average='weighted'))
print('New F1 score %s' % metrics.f1_score(y_pred, y_test,average='weighted'))

# %% In [51]:
# Saving each metric to add to a dictionary for logging
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Create dictionaries for the data we want to log          
metrics = {"f1": f1,
           "recall": recall,
           "precision": precision}

params= {'classifier': 'linear SVC',
         'max_df': 0.8,
         'min_df': 2,
         'ngram_range': '(1,2)',
         'vectorizer': 'Tfidf',
         'scaling': 'no',
         'resampling': 'no',
         'test_train random state': '0'}
  
# Log info on comet
experiment.log_metrics(metrics)
experiment.log_parameters(params)

# End experiment
experiment.end()

# Display results on comet page
experiment.display()

