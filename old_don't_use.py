from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE
import numpy as np
import os
import string


def read_data(folder_name = "deceptive_from_MTurk"):
    """
    This one returns a dic of fold - reviews, where reviews are a list of strings 
    """
    corpora = {}
    path_to_foulder = "op_spam_v1.4/negative_polarity/" + folder_name
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

def most_frequent_words(text, N, grams = 1):
    count_vec = CountVectorizer(stop_words = "english", ngram_range = (grams, grams)).fit([text])
    X_count = count_vec.transform([text])
    words_freq = [(word, X_count[0, idx]) for word, idx in count_vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:N]

def all_words_occuring_once(text, grams = 1):
    count_vec = CountVectorizer(ngram_range = (grams, grams)).fit([text])
    X_count = count_vec.transform([text])
    words_freq = [(word, X_count[0, idx]) for word, idx in count_vec.vocabulary_.items()]
    single_words = [word for (word, count) in words_freq if count == 1]
    return single_words
    
def average_count_punctuation(corpora):
    punctuation_count = sum(1 for review in corpora for char in review if char in string.punctuation)
    return punctuation_count/len(corpora)

if __name__ == "__main__":
    true_corpora = read_data("truthful_from_Web")
    fake_corpora = read_data()

    ### SANITY CHECK - PASSED
    # print(len(true_corpora))
    # print(len(true_corpora['fold3']))

    # print(len(fake_corpora))
    # print(len(fake_corpora['fold4']))

    all_text_from_true_corpora = sum(list(true_corpora.values()), []) # this is a list of strings
    all_text_from_true_corpora_flatten = " ".join(all_text_from_true_corpora)
    all_text_from_fake_corpora = sum(list(fake_corpora.values()), []) # this is a list of strings
    all_text_from_fake_corpora_flatten = " ".join(all_text_from_fake_corpora)

    lens_true_corpora = [len(review.split(" ")) for review in all_text_from_true_corpora]
    avg_len_true_corpora = sum(lens_true_corpora)/len(lens_true_corpora)

    lens_fake_corpora = [len(review.split(" ")) for review in all_text_from_fake_corpora]
    avg_len_fake_corpora = sum(lens_fake_corpora)/len(lens_fake_corpora)

    # print("\n ################################################################## \n")
    # print("The average length of a true review is : ", avg_len_true_corpora, "\n\
    #        where as a fake one has an average of ", avg_len_fake_corpora)
    
    # print("\n ################################################################## \n")
    # print("The average punctuation in a true review is : ", average_count_punctuation(all_text_from_true_corpora), "\n\
    #         where for a fake one is ", average_count_punctuation(all_text_from_fake_corpora))
    
    # ### finding the most common N words : 
    # print("\n ################################################################## \n")
    # print("The 20 most frequent words from the true corpora are: ")
    # print(most_frequent_words(all_text_from_true_corpora_flatten, 20, 1))

    # print("The 20 most frequent words from the fake corpora are: ")
    # print(most_frequent_words(all_text_from_fake_corpora_flatten, 20, 1))

    # print("\n ################################################################## \n")
    # print("The 20 most frequent bigrams from the true corpora are: ")
    # print(most_frequent_words(all_text_from_true_corpora_flatten, 20, 2))
    
    # print("The 20 most frequent bigrams from the fake corpora are: ")
    # print(most_frequent_words(all_text_from_fake_corpora_flatten, 20, 2))

    # print("\n ################################################################## \n")
    # print("The 20 most frequent trigrams from the true corpora are: ")
    # print(most_frequent_words(all_text_from_true_corpora_flatten, 20, 3))

    # print("The 20 most frequent trigrams from the fake corpora are: ")
    # print(most_frequent_words(all_text_from_fake_corpora_flatten, 20, 3))

    print("\n ################################################################## \n")
    print("The words that appear just once from the true corpora are: ")
    print(all_words_occuring_once(all_text_from_true_corpora_flatten, 1))

    print("The words that appear just once from the fake corpora are: ")
    print(all_words_occuring_once(all_text_from_fake_corpora_flatten, 1))

