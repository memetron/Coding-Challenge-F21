import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

SA = SentimentIntensityAnalyzer()
print(SA.polarity_scores(open("input.txt","r").read()))