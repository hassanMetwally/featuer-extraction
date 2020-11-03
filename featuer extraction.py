
from nltk.corpus import movie_reviews

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd



#print(movie_reviews.fileids())


def func(list0):
    sents = []
    for blist in list0:
        fsent = ""
        for slist in blist:
            fsent += slist + " "
        sents.append(fsent)
    return sents


sents0 = movie_reviews.sents("neg/cv000_29416.txt")
sents1 = movie_reviews.sents("pos/cv041_21113.txt")

texts0 = func(sents0)
texts1 = func(sents1)

texts2 = texts0 + texts1


vec = CountVectorizer()
vec.fit(texts2)

print([w for w in sorted(vec.vocabulary_.keys())])

print(pd.DataFrame(vec.transform(texts2).toarray(),columns=sorted(vec.vocabulary_.keys())))
