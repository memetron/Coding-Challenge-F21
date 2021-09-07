import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

#Creates SentimentAnalyzer
SA = SentimentIntensityAnalyzer()

#Finds polarity scores of overall text
text = open("input.txt","r").read().replace("\n"," ")
print("Overall score: " + str(SA.polarity_scores(text)))

#Finds polarity scores of individual sentences
scores = []
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = tokenizer.tokenize(text)
print("\n"+"-"*100+"\nSentence scores: \n")
for sentence in sentences:
    score = SA.polarity_scores(sentence)
    print(sentence + "\n" + str(score)+"\n")
    scores.append(score)


pos = []
neg = []
compound = []
#Graphs scores
for score in scores:
    pos.append(score["pos"])
    neg.append(score["neg"])
    compound.append(score["compound"])
#compound score
plt.plot(compound)
plt.ylabel("compound score")
plt.show()
#negative score
plt.plot(neg)
plt.ylabel("negative score")
plt.show()
#positive score
plt.plot(pos)
plt.ylabel("positive score")
plt.show()