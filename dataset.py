from main import *
import re

def custom_preprocessor(text):

    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    tokens = text.split()
    # Remove personalized stop words
    tokens = [word for word in tokens if word.lower() not in infrequent_words]
    return ' '.join(tokens)

def separate_dataset(real = True):
    if real : 
        reviews = read_data("truthful_from_Web")
    else: 
        reviews = read_data()

    test = reviews.pop('fold5')
    train = sum(list(reviews.values()), [])   
    
    return train, test


def make_n_gram_dataset(n_min, n_max):
    

    count_vec = CountVectorizer(stop_words = 'english', preprocessor=custom_preprocessor, ngram_range = (n_min, n_max)).fit([flat_train_true, flat_train_fake])
    
    X_train_true = count_vec.transform(train_true)
    Y_train_true = np.ones(X_train_true.shape[0])

    X_test_true = count_vec.transform(test_true)
    Y_test_true = np.ones(X_test_true.shape[0])

    ##################

    X_train_fake = count_vec.transform(train_fake)
    Y_train_fake = np.zeros(X_train_fake.shape[0])

    X_test_fake = count_vec.transform(test_fake)
    Y_test_fake = np.zeros(X_test_fake.shape[0])

    ############## Combining them 

    X_train = np.vstack((X_train_true.toarray(), X_train_fake.toarray()))
    Y_train = np.hstack((Y_train_true, Y_train_fake))
    X_test = np.vstack((X_test_true.toarray(), X_test_fake.toarray()))
    Y_test = np.hstack((Y_test_true, Y_test_fake))

    return X_train, Y_train, X_test, Y_test, count_vec.get_feature_names_out()



train_true, test_true = separate_dataset()
flat_train_true = " ".join(train_true)
flat_test_true = " ".join(test_true)

train_fake, test_fake = separate_dataset(real = False)
flat_train_fake = " ".join(train_fake)
flat_test_fake = " ".join(test_fake)

infrequent_words = all_words_occuring_once(flat_train_true + flat_train_fake)

X_train, Y_train, X_test, Y_test, feature_names = make_n_gram_dataset(1, 2)
print(X_train.shape)
print(feature_names)


