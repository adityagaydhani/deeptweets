"""
This code adapted from https://gist.github.com/yanofsky/5436496.
"""

import json
import sys
import re

import pandas as pd
import tweepy  # https://github.com/tweepy/tweepy

# Twitter API credentials
with open('twitter_keys.json', 'r') as f:
    keys = json.load(f)

consumer_key = keys['api_key']
consumer_secret = keys['api_secret_key']
access_token = keys['access_token']
access_token_secret = keys['access_token_secret']


def get_all_tweets(screen_name):
    try:
        df = pd.read_csv(f'../data/{screen_name}.csv')
        since_id = df['id'][0]
        print(since_id)
        print(type(since_id))
    except:
        since_id = 1

    # Twitter only allows access to a users most recent 3200 tweets with
    # this method

    # authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # initialize a list to hold all the tweepy Tweets
    all_tweets = []

    # make initial request for most recent tweets (200 is the maximum
    # allowed count)
    new_tweets = [[tweet.id, tweet.full_text] for tweet in api.user_timeline(
        screen_name=screen_name, count=200, tweet_mode='extended',
        since_id=since_id)]

    # save most recent tweets
    all_tweets.extend(new_tweets)

    if not all_tweets:
        return all_tweets
    # save the id of the oldest tweet less one
    oldest = all_tweets[-1][0] - 1

    # keep grabbing tweets until there are no tweets left to grab
    old_length = len(all_tweets)
    for _ in range(5):
        while True:
            print(f'getting tweets before {oldest}')

            # all subsiquent requests use the max_id param to prevent
            # duplicates
            new_tweets = [[tweet.id, tweet.full_text] for tweet in api.user_timeline(
                screen_name=screen_name, count=200, max_id=oldest,
                since_id=since_id, tweet_mode='extended')]

            # save most recent tweets
            all_tweets.extend(new_tweets)

            # update the id of the oldest tweet less one
            oldest = all_tweets[-1][0] - 1

            print(f'...{len(all_tweets)} tweets downloaded so far')

            if len(all_tweets) == old_length:
                break
            else:
                old_length = len(all_tweets)

    return all_tweets


def preprocess(text):
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')

    bad_start = ['http:', 'https:']
    for w in bad_start:
        text = re.sub(f" {w}\\S+", "", text)
        text = re.sub(f"{w}\\S+ ", "", text)
        text = re.sub(f"\n{w}\\S+ ", "", text)
        text = re.sub(f"\n{w}\\S+", "", text)
        text = re.sub(f"{w}\\S+", "", text)
    text = re.sub(' +', ' ', text)
    text = ' '.join(text.split())
    text = text.strip()

    expressions = ['http', '@', '#']
    not_boring_words = len([None for w in text.split() if all(
        e not in w.lower() for e in expressions)])
    
    if not_boring_words < 3:
        return ''
    return text

if __name__ == '__main__':
    screen_name = sys.argv[1]
    all_tweets = get_all_tweets(screen_name)
    print(len(all_tweets))
    # Convert to Pandas DataFrame
    df = pd.DataFrame(all_tweets, columns=['id', 'tweet'])

    # Remove retweets
    df = df[~df.tweet.str.contains(r'RT @')]

    # Remove URLs from tweets
    df['tweet'] = df.tweet.str.replace(r'http\S+|www.\S+', '', case=False)

    # Drop NA
    df = df.dropna()

    # Remove tweets with less than 4 words
    df = df[~(df.tweet.apply(lambda x: len(x.split())) < 4)]

    print(len(df))

    try:
        original_df = pd.read_csv(f'../data/{screen_name}.csv')
    except:
        original_df = pd.DataFrame()

    df = pd.concat([df, original_df]).reset_index(
        drop=True)[['id', 'tweet']]

    df.tweet = df.tweet.apply(preprocess)
    df = df[df.tweet != '']
    df.to_csv(f'../data/{screen_name}.csv', index=False)
