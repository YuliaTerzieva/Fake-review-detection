from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE
import numpy as np
# import spacy
import os


def read_data(folder_name = "deceptive_from_MTurk"):
    corpora = {}
    path_to_foulder = "Fake-review-detection/op_spam_v1.4/negative_polarity/" + folder_name
  
    # Change the directory 
    for file in os.listdir(path_to_foulder): 
        fold = []
        for text_file in os.listdir(path_to_foulder + "/"+file): 
            # Check whether file is in text format or not 
            if text_file.endswith(".txt"):
                f = open(path_to_foulder + "/"+file+"/"+text_file, "r")
                fold.append(f.read())
            # print("I just read ", text_file, "from ", file)
        corpora[file] = fold
        # print("I just read file ", file)

    return corpora

if __name__ == "__main__":
    true_corpora = read_data("truthful_from_Web")
    fake_corpora = read_data()

    print(len(true_corpora))
    print(len(true_corpora['fold3']))

    print(len(fake_corpora))
    print(len(fake_corpora['fold4']))
