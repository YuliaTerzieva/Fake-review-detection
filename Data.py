from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import string
import re
import os

class Data :
    def __init__(self, max_grams):
        self.train_true, self.test_true = self.separate_dataset()
        self.flat_train_true = " ".join(self.train_true)

        self.train_fake, self.test_fake = self.separate_dataset(real = False)
        self.flat_train_fake = " ".join(self.train_fake)

        self.infrequent_words = self.all_words_occuring_once(self.flat_train_true + self.flat_train_fake)

        self.X_train, self.Y_train, self.X_test, self.Y_test, self.feature_names = self.make_n_gram_dataset(1, max_grams)


    def read_data(self, folder_name = "deceptive_from_MTurk"):
        """
        A funtion that take a folder name and creates a dictionary with 
        kays the folds and value the reviews in a list. 
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

    def most_frequent_words(self, text, N, grams = 1):
        """
        params: 
                - text  - a string of all the interviews 
                - N     - a number indicating how many words should the program returns
                - grams - unigrams or bigrams or whatever you want out
        """
        count_vec = CountVectorizer(stop_words = "english", ngram_range = (grams, grams)).fit([text])
        X_count = count_vec.transform([text])
        words_freq = [(word, X_count[0, idx]) for word, idx in count_vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:N]

    def all_words_occuring_once(self, text, grams = 1):
        count_vec = CountVectorizer(ngram_range = (grams, grams)).fit([text])
        X_count = count_vec.transform([text])
        words_freq = [(word, X_count[0, idx]) for word, idx in count_vec.vocabulary_.items()]
        single_words = [word for (word, count) in words_freq if count == 1]
        return single_words
        
    def average_count_punctuation(self, corpora):
        punctuation_count = sum(1 for review in corpora for char in review if char in string.punctuation)
        return punctuation_count/len(corpora)

    def custom_preprocessor(self, text):
        """
            This is a function to customly filter/preprocess the text before it is used for vetorization/feature selection
            It is used in CountVectorizer as a callable function
        """

        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        
        tokens = text.split()
        # Remove personalized stop words
        tokens = [word for word in tokens if word.lower() not in self.infrequent_words]
        return ' '.join(tokens)

    def separate_dataset(self, real = True):
        """
            This functions creates two lists from the dictinary with fold created in read_data
            One list for training containing folds 1 to 4 and one list for testing containing just fold 5
        """
        if real : 
            reviews = self.read_data("truthful_from_Web")
        else: 
            reviews = self.read_data()

        test = reviews.pop('fold5')
        train = sum(list(reviews.values()), [])   
        
        return train, test

    def make_n_gram_dataset(self, n_min, n_max):
        
        count_vec = CountVectorizer(stop_words = 'english', preprocessor=self.custom_preprocessor, ngram_range = (n_min, n_max))
        count_vec = count_vec.fit([self.flat_train_true, self.flat_train_fake])
        
        X_train_true = count_vec.transform(self.train_true)
        Y_train_true = np.ones(X_train_true.shape[0])

        X_test_true = count_vec.transform(self.test_true)
        Y_test_true = np.ones(X_test_true.shape[0])

        ##################

        X_train_fake = count_vec.transform(self.train_fake)
        Y_train_fake = np.zeros(X_train_fake.shape[0])

        X_test_fake = count_vec.transform(self.test_fake)
        Y_test_fake = np.zeros(X_test_fake.shape[0])

        ############## Combining them 

        X_train = np.vstack((X_train_true.toarray(), X_train_fake.toarray()))
        Y_train = np.hstack((Y_train_true, Y_train_fake))
        X_test = np.vstack((X_test_true.toarray(), X_test_fake.toarray()))
        Y_test = np.hstack((Y_test_true, Y_test_fake))

        return X_train, Y_train, X_test, Y_test, count_vec.get_feature_names_out()
