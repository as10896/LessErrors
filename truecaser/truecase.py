import os
from .Truecaser import *
import pickle
import nltk
import string

basename = os.path.dirname(os.path.realpath(__file__))

print("Load Truecaser.....", end="")
with open(os.path.join(basename, 'distributions.obj'), 'rb') as f:
    uniDist = pickle.load(f)
    backwardBiDist = pickle.load(f)
    forwardBiDist = pickle.load(f)
    trigramDist = pickle.load(f)
    wordCasingLookup = pickle.load(f)
print("Done.")

def truecase(sent):
    tokensCorrect = nltk.word_tokenize(sent)
    tokens = [token.lower() for token in tokensCorrect]
    tokensTrueCase = getTrueCase(tokens, 'title', wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
    return " ".join(tokensTrueCase)