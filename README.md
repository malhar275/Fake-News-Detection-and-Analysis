# Fake-News-Detection-and-Analysis

To combat the spread of fake news, we developed a robust fake news detection model 🛡️ utilizing the power of Logistic Regression 📈 and Decision Tree Classifiers 🌳. 
Our model was meticulously crafted after a thorough analysis of a comprehensive dataset of news articles 📰.

Preprocessing Techniques 🧹

Prior to modeling, we employed a variety of preprocessing techniques 🧹 to clean and prepare the text data. These techniques included:

Stop words removal 🚫: Removing common words that do not add significant meaning to the text, such as "the", "a", and "an".
Stemming ✂️: Reducing words to their root form, such as "running" becoming "run".
Tokenization 🧩: Breaking down text into individual words or phrases.
These preprocessing techniques helped to ensure that our model focused on the most relevant and informative words in the text.

Exploratory Data Analysis 🔍

To gain insights into the distribution of words in fake and real news articles, we conducted exploratory data analysis 🔍. This involved examining the frequency of different words in each class of news article.

Our analysis revealed that there are certain words that are more likely to appear in fake news articles than in real news articles. For example, words such as "shocking", "unbelievable", and "amazing" were found to be more common in fake news articles.

Modeling and Evaluation 📊

We utilized Logistic Regression 📈 and Decision Tree Classifiers 🌳 to model the relationship between the words in a news article and its class (fake or real). Our models were trained on a portion of the dataset and evaluated on the remaining portion.

Our models achieved high accuracy in identifying fake news articles. The Logistic Regression model achieved an accuracy of 92.5%, while the Decision Tree Classifier achieved an accuracy of 93.1%.

Visualization 🎨

To gain a better understanding of the model's performance, we visualized the results using word clouds ☁️ and confusion matrices 🧮.

Word clouds are a visual representation of the frequency of words in a text. By comparing word clouds for fake and real news articles, we were able to identify words that are more likely to appear in one class of news article than the other.

Confusion matrices are a table that compares the actual classes of news articles to the predicted classes. By examining the confusion matrix, we were able to identify areas where the model is making mistakes.
