# IndependentStudy

## Feature Analysis
#### 1. Sentiment score

Here I have calculated positive and negative sentiment for each feedback using SentimentIntensityAnalyzer which uses [vader](http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf) algorithm to compute the sentiment score. The sentiment score for each type of sentence is plotted on a scatter plot. The result is in the image shown below.

![sentiment_result](https://github.com/UtkarshVIT/independentstudy/blob/master/images/pos_tags.png "Logo Title Text 1")

#### 2. Part of speech ratio

Here I calculated the ratio of count of each POS word to the total words in a sentence using nltk.pos_tag() method. The ratio each pos tag for each sentence is plotted on a scatter plot. The result is shown below in the image.

![sentiment_result](https://github.com/UtkarshVIT/independentstudy/blob/master/images/sentiment.png "Logo Title Text 1")
