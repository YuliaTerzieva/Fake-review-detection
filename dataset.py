from main import *

def separate_dataset(real = True):
    if real : 
        reviews = read_data("truthful_from_Web")
    else: 
        reviews = read_data()

    test = reviews.pop('fold5')
    train = sum(list(reviews.values()), [])   
    
    return train, test


def make_unigram_dataset():
    train_true, test_true = separate_dataset()
    flat_train_true = " ".join(train_true)
    flat_test_true = " ".join(test_true)

    count_vec_true = CountVectorizer(stop_words = "english", ngram_range = (1, 1)).fit([flat_train_true])
    
    X_train_true = count_vec_true.transform(train_true)
    Y_train_true = np.ones(X_train_true.shape[0])

    X_test_true = count_vec_true.transform(test_true)
    Y_test_true = np.ones(X_test_true.shape[0])

    ##################

    train_fake, test_fake = separate_dataset(real = False)
    flat_train_fake = " ".join(train_fake)
    flat_test_fake = " ".join(test_fake)

    count_vec_fake = CountVectorizer(stop_words = "english", ngram_range = (1, 1)).fit([flat_train_fake])
    
    X_train_fake = count_vec_fake.transform(train_fake)
    Y_train_fake = np.zeros(X_train_fake.shape[0])

    X_test_fake = count_vec_fake.transform(test_fake)
    Y_test_fake = np.zeros(X_test_fake.shape[0])

    ############## Combining them 

    X_train = np.vstack(X_train_true, X_train_fake)
    Y_train = np.vstack(Y_train_true, Y_train_fake)
    X_test = np.vstack(X_test_true, X_test_fake)
    Y_test = np.vstack(Y_test_true, Y_test_fake)

    return X_train, Y_train, X_test, Y_test




if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = make_unigram_dataset()


