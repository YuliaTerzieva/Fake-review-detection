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

def get_features(text):
    count_vec = CountVectorizer(stop_words = "english")
    X_count = count_vec.fit_transform(text)
    print('CountVectorizer:')
    print(count_vec.get_feature_names_out()[300:310])
    print(X_count.toarray()[0][300:310]) 

    tfidf_vec = TfidfVectorizer()
    X_tfidf = tfidf_vec.fit_transform(text)
    print('TF-IDF:')
    print(tfidf_vec.get_feature_names_out()[300:310])
    print(X_tfidf.toarray()[0][300:310])

def most_frequent_words(text, N):
    count_vec = CountVectorizer(stop_words = "english")
    X_count = count_vec.fit_transform([text])
    print(X_count.shape)
    # this is a 1d array - find the indices of the N largest numbers and then find what words those are in get_feature_names_out()
    print('CountVectorizer:')
    print(count_vec.get_feature_names_out()[0:10])
    print(X_count.toarray()[0][0:10]) 

if __name__ == "__main__":
    true_corpora = read_data("truthful_from_Web")
    fake_corpora = read_data()

    ### SANITY CHECK - PASSED
    # print(len(true_corpora))
    # print(len(true_corpora['fold3']))

    # print(len(fake_corpora))
    # print(len(fake_corpora['fold4']))

    all_text_from_true_corpora = sum(list(true_corpora.values()), []) # this is a list of strings
    # get_features(all_text_from_true_corpora)

    ## The results are not that good. we need to do sth else, this is houwing the documtne-term matrix i.e. in each review how many time does each word occur. This is not useful. 

    ### finding the most common N words : 
    all_text_from_true_corpora_flatten = " ".join(all_text_from_true_corpora)
    most_frequent_words(all_text_from_true_corpora_flatten, 10)
    '''
    Suggested features 
     - len of review
     - how many dots, columns, sth are used, normalized by the length
     - 1) we can find the most frequent words in the corpora and then 2) see how many words in the review are in the 10/20/30 most frequent words used (also normalized by length?)
     '''





