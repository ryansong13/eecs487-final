import re
import string
import torch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.casual import WORD_RE
import numpy as np
from collections import defaultdict
"""
NOTE: code from https://towardsdatascience.com/sentiment-analysis-using-logistic-regression-and-naive-bayes-16b806eb4c4b
"""

# Preprocessing tweets
def process_tweet(tweet, party):
    #Remove old style retweet text "RT"
    #tweet2 = re.sub(r'^RT[\s]','', tweet) # not sure if we need this
    
    # Remove hyperlinks
    tweet2 = re.sub(r'https?:\/\/.*[\r\n]*','', tweet)
    
    # Remove hastag # symbol
    tweet2 = re.sub(r'#','',tweet2)
        
    # TODO: look into different settings for TweetTokenizer
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet2) 

    stopwords_english = stopwords.words('english') 
    
    tweets_clean = []
    for word in tweet_tokens:
        if word not in stopwords_english and word not in string.punctuation:
            tweets_clean.append(word)

    stemmer = PorterStemmer()
    freqs = defaultdict(int) 
    tweets_stem = []
    for word in tweets_clean:
        stem_word = stemmer.stem(word) 
        tweets_stem.append(stem_word)
        pair = (stem_word, party)
        freqs[pair] += 1 
    # UNK tokens

    # print(tweets_stem)
    return freqs, tweets_stem # freqs only for training, tweets_stem for train+test

def handle_UNK(freqs, party):

  k = ("UNK", party)

  for key in list(freqs):
    if freqs[key] < 2:

      freqs.pop(key)

      if k in freqs:
        freqs[k] += 1
      else:
        freqs[k] = 1
# TODO: explore bigrams/ngrams as a feature

def extract_features(tweet, freqs):
    '''
    Input: 
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    '''
    # process_tweet tokenizes, stems, and removes stopwords
    #word_l = process_tweet(tweet, -1)
    
    x = np.zeros((1, 3)) 
    #bias term is set to 1
    x[0,0] = 1 
      
    for word in tweet:
        # increment the word count for the republican label 1
        x[0,1] += freqs.get((word,1),0)
        
        # increment the word count for the democratic label 0
        x[0,2] += freqs.get((word,0),0)
        
    assert(x.shape == (1, 3))
    return x

def embed_features(data, embed):
  
  max_len = max([len(d) for d in data])
  output = torch.zeros((len(data), max_len, 300))
  
  i = 0
  for tweet in data:
    l = torch.zeros((max_len, 300))
    idx = 0
    for word in tweet: 
      # NOTE: max_len dimension isn't consistent across tweets?
      """
      eg:
      tweet1: "vote for biden"
      tweet2: "universal healthcare"
      
      our output would be 2 (num_tweets) x 3 (max_len) x 300 (word embed)
      but the first col is "vote" for 1st tweet and "universal" for second
      
      or should max_len dimension be |vocab| and then one-hot encoding -- no need for padding then
      """
      if word in embed:
        l[idx] = torch.from_numpy(embed[word])
        idx += 1
    output[i] = l
    i += 1
  print("one tweet shape:", output[0].shape) # should be (max_len, 300)
  print("output shape:", output.shape)
  return output