# IndependentStudy

## Feature Analysis
#### 1. Sentiment score

Here I have calculated positive and negative sentiment for each feedback using SentimentIntensityAnalyzer which uses [vader](http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf) algorithm to compute the sentiment score. The sentiment score for each type of sentence is plotted on a scatter plot. The result is in the image shown below.

![sentiment_result](https://github.com/UtkarshVIT/independentstudy/blob/master/images/pos_tags.png "Logo Title Text 1")

#### 2. Part of speech ratio

Here I calculated the ratio of count of each POS word to the total words in a sentence using nltk.pos_tag() method. The ratio each pos tag for each sentence is plotted on a scatter plot. The result is shown below in the image.

![sentiment_result](https://github.com/UtkarshVIT/independentstudy/blob/master/images/sentiment.png "Logo Title Text 1")

#### 3. Words as tf-idf

Here is distributed the data 85% - 15% into training and test set and created a document vector using the training set. I created two documents for 'sentences with problems' and 'sentences without problems'. I computed the sum of the tf-idf value for each word in a sentence using the document vector and classified the sentence into the document for which it had the max sum.

This process could classify **70%** of the sentences correctly.

