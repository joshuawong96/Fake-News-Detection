# Fake-News-Detection
This project on Fake News Detection using Data Mining and Machine Learning techniques was one of the assessment components for the Singapore Management University (SMU) module IS434 - Data Mining and Business Analytics.

---

## Table of Contents
- [Introduction](#Introduction)
    - [Motivation](#Motivation)
- [Dataset](#Dataset)
    - [Data Preprocessing](#Data-Preprocessing)
- [Modelling](#Modelling)
    - [Feature Engineering ](#Feature-Engineering)
    - [Word Representation Models](#Word-Representation-Models)
    - [Classification Modelling](#Classification-Modelling)
    - [Hyperparameter Tuning](#Hyperparameter-Tuning)
- [Findings & Insights](#Findings-and-Insights)
- [Code Navigation](#Code-Navigation)
- [Contributors](#contributors)

---

## Introduction

In this project, the team aims to use Natural Language Processing and Machine Learning techniques to build an effective classifier that can detect fake news. 

The rampant spread of fake news via online news media coupled with limitations of current manual fake news detection methods means there is a pressing need to develop more effective solutions to tackle the problem.
<br>

### Motivation
Extensive spread of fake news has the potential for extremely negative impacts on individuals and society. Fake news is very difficult to tackle, especially with the rampant spread of fake news due to the digitalisation of news. 

In recent years, consumers are constantly bombarded with media online, without a choice in picking the news they choose to read. With tons of news sources flooding the internet, discerning between real and fake news content online is not as simple a task for the average news consumer.
<br>

[Back To The Top](#top)

---
## Dataset
For this project, 2 datasets were combined: [Fake and real news dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) and [Fake News](https://www.kaggle.com/c/fake-news/data?select=train.csv).
Both datasets are open-sourced and can be found on Kaggle in the respective links. 

A total of 65,101 news articles will be used for modelling, of which 23,481 are true news and 21,417 are fake news.
<br>

### Data Preprocessing
Fake news are encoded as 1 (positive) class and true news as 0 (negative class), to standardise the labels across both datasets. Title of the articles were joined with article bodies to form the corpus. Redundant columns such as the author, date of publish and subject were dropped. Punctuations were removed, text was converted to lowercase, tokenized, stemmed and stopwords were remoed using regex and ntlk packages.

[Back To The Top](#top)

---
## Modelling
This section will elaborate of the different models and techniques used. The diagram below shows the modelling workflow for this classification problem. 

[![Modelling-Workflow.png](https://i.postimg.cc/JnMc5MHz/Modelling-Workflow.png)](https://postimg.cc/06Xw8gCT)
<br>

### Feature Engineering 

Features that describe the characteristics of each article were generated to add a new dimension to the data in addition to analysing text data. These features were hypothesised to have significant differences between the 2 classes and according to popular literature, would be useful in classification. 

Features created are as follows:
- Number of words
- Number of sentences
- Number of characters
- Proportion of punctuations
- Proportion of nouns
- Proportion of verbs
- Proportion of adjectives
- Proportion of words in quotes
- Average sentence length
- Proportion of unique words 
- Proportion of stopwords
- Proportion of discourse relations 
- Sentiment polarity
<br>

### Word Representation Models
Different word representation models were tried to uncover which model would give higher performance. 

CountVectorizer was a Bag of words approach to transform text into a vector based on the number of occurrences of each word. TF-IDF Vectorizer focuses not only on frequency of words but also penalises words that occur too often in the corpus, which hold less meaning in differentiating text between true and fake news. 

A self-trained Word2Vec model was also tested out but did not perform as well as traditional word representation models. 

### Classification Modelling 
Various classification models were tested to find the most optimal model.

For single classifiers:
- Naive Bayes (Baseline model)
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree

For ensemble methods:
- Hard Voting Classifier with Naive Bayes, Logistic Regression, SVM and Decision Tree
- Random Forest
- AdaBoost
- XGBoost
<br>

### Hyperparameter Tuning 
GridSearchCV was used on multiple models (within computational resources constraint) to find an optimal set of hyperparameters instead of manual trial and error. 

[Back To The Top](#top)

---
## Findings and Insights
- Based on our evaluation metrics of Precision, Recall and F1 score, XGBoost performed the best. The baseline model Naive Bayes performed the worst and tuned models performed better than base models for Logistic Regression, SVM and Decision Tree. Ensemble methods generally performed better. 
- Added features unexpectedly did not have much use except for XGBoost and more work can be done to explore other features. Boxplots also seemed to serve as a relatively good indicator of feature importance.

[Back To The Top](#top)

---
## Code Navigation
1. Data Folder - contains datasets (2 original datasets, 1 combined dataset, 1 processed dataset)
2. Data Preparation Folder - contains notebooks for preprocessing, EDA, exploring different word representation mdoels, feature engineering and feature selection 
3. Analysis Folder - contains notebooks for classification models 

[Back To The Top](#top)

---
## Contributors

1. Belle Tan Xuan Ting ([Github](https://github.com/bellebasaur))
2. Joshua Wong Yeung Nguon ([Github](https://github.com/joshuawong96))
3. Lai Hui Jing ([Github](https://github.com/huijingg))
4. Mabelle Tham Shi Qin ([Github](https://github.com/mabelletham))
5. Ow Ling Jia ([Github](https://github.com/owlingjia))

[Back To The Top](#top)
