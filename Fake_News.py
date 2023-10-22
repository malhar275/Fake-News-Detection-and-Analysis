#!/usr/bin/env python
# coding: utf-8

# In[5]:


# importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


data = pd.read_csv('News.csv',index_col=0) 
#index_col=0 specify the first column of the CSV file should be used as the index of the dataframe
data.head()


# In[25]:


data.shape


# In[4]:


# As the title, subject and date column will not going to be helpful in identification of the news. 
# So, we can drop these column
# data.drop(["title", "subject","date"], axis = 1)
data = data.drop(["title", "subject","date"], axis = 1)


# In[5]:


data.head()


# In[60]:


# df=['january','february','march','april','may','june','july','august','september','october','november','december']
data['date']

data.dropna(axis=0, inplace=True)
data.isnull().sum()


# In[71]:


# checking for null values
t = []
for i in data['date']:
    f=str(i).split(" ")
    t.append(f[0])
# print(t)
data['Month'] = pd.DataFrame(t)
data['Month'] = data['date'].dt.month_name()


# In[66]:


data.sample(5)


# In[7]:


# Shuffling,resting the index
data = data.sample(frac=1)
data.reset_index(inplace=True)
data.drop(["index"], axis=1, inplace=True)


# In[8]:


data.head()


# In[12]:


import seaborn as sns
sns.countplot(data=data,x='class',order=data['class'].value_counts().index)


# In[10]:


gypip install tqdm


# In[ ]:





# In[11]:


get_ipython().system('pip install wordcloud')


# In[12]:


from tqdm import tqdm
import re
import nltk 
nltk.download('punkt') 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
# str="This is a sentence"
# tokenized_word=["This","is","a","sentence"]


# In[13]:


#Function for preprocess of text
def preprocess_text(text_data):
    # take empty list
    preprocessed_text = []
    for sentence in tqdm(text_data):
        # re.sub() to remove all punctuation from the sentence
        sentence = re.sub(r'[^\w\s]', '', sentence)
        preprocessed_text.append(' '.join(token.lower()
                                          for token in str(sentence).split()
                                          if token not in stopwords.words('english')))

    return preprocessed_text


# In[14]:


preprocessed_review = preprocess_text(data['text'].values)
data['text'] = preprocessed_review


# In[15]:


# Real
consolidated = ' '.join(word for word in data['text'][data['class'] == 1].astype(str))
wordCloud = WordCloud(width=1600,height=800,random_state=21,max_font_size=110,collocations=False)
plt.figure(figsize=(15, 10))
plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')
plt.axis('off')
plt.show()


# In[17]:


# Fake
consolidated = ' '.join(word for word in data['text'][data['class'] == 0].astype(str))
wordCloud = WordCloud(width=1600,height=800,random_state=21,max_font_size=110,collocations=False)
plt.figure(figsize=(15, 10))
plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')
plt.axis('off')
plt.show()


# In[21]:


from sklearn.feature_extraction.text import CountVectorizer


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1],reverse=True)
    return words_freq[:n]


common_words = get_top_n_words(data['text'], 20)
df1 = pd.DataFrame(common_words, columns=['Review', 'count'])

df1.groupby('Review').sum()['count'].sort_values(ascending=False).plot(kind='bar',figsize=(10, 6),xlabel="Top Words",ylabel="Count",title="Bar Chart of Top Words Frequency")


# In[14]:


# split a dataset into train and test sets.
from sklearn.model_selection import train_test_split
# to calculate the accuracy of a model on a given dataset
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# first two arguments to the train_test_split() function are the features and target variables of the 
# dataset, respectively. The third argument is the test size, which is the proportion of the dataset 
# that should be allocated to the test set. In this case, the test size is 25%, so 75% of the dataset 
# will be allocated to the train set and 25% will be allocated to the test set.

x_train, x_test, y_train, y_test = train_test_split(data['text'],data['class'],test_size=0.25)


# In[15]:


# use to vectorize the text data in the train and test sets.
# Vectorization is the process of converting text data into a numerical representation.
from sklearn.feature_extraction.text import TfidfVectorizer

# term frequency inverse deocument frequency
vectorization = TfidfVectorizer() # create a object
x_train = vectorization.fit_transform(x_train) #fit the vectorize to the train set
x_test = vectorization.transform(x_test) # transfor train sets using vectorizer

# This converts the text data in the test set into a numerical representation.


# In[23]:


from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression() #creates a LogisticRegression object.
model1.fit(x_train, y_train) # trains the model on test set

# calculate accuracy on train and test set
train_accuracy = accuracy_score(y_train, model1.predict(x_train))
test_accuracy = accuracy_score(y_test, model1.predict(x_test))

# testing the model
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)


# In[17]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# testing the model
train_accuracy = accuracy_score(y_train, model.predict(x_train))
test_accuracy = accuracy_score(y_test, model.predict(x_test))

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)


# In[18]:


# Confusion matrix of Results from Decision Tree classification
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, model.predict(x_test))

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[False, True])

cm_display.plot()
plt.show()


# In[72]:


# Confusion matrix of Results from LogisticRegression
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, model1.predict(x_test))

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[False, True])

cm_display.plot()
plt.show()


# In[73]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you already have your DataFrame named 'data'

# Convert the 'date' column to datetime and extract the month
data['date'] = pd.to_datetime(data['date'], format='%B %d, %Y', errors='coerce')
data['month'] = data['date'].dt.month

plt.scatter(data['month'], data['text'], alpha=0.5)
plt.xlabel('Month')
plt.ylabel('Class')
plt.title('Scatter Plot of Month vs. Class')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)


plt.show()

