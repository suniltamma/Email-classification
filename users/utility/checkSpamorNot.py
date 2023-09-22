# import the libraries
import sys

assert sys.version_info >= (3, 5)

# data manipulation
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
# %matplotlib inline
from django.conf import settings
# consistent sized plot
from pylab import rcParams

rcParams['figure.figsize'] = 12, 5
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['axes.labelsize'] = 12

# display options for dataframe
pd.options.display.max_columns = None

# text processing
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
# from fuzzywuzzy import process
# from fuzzywuzzy import fuzz

# text feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# models
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

# regular expressions

# string operations
import string

# compute articles similarity
from sklearn.metrics import confusion_matrix

# for file extraction

# ignore warnings
import warnings

warnings.filterwarnings(action='ignore', message='')
stemmer = PorterStemmer()


def simplify(text):
    '''Function to handle the diacritics in the text'''
    import unicodedata
    try:
        text = text  # unicode(text, 'utf-8')
    except NameError:
        pass
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return str(text)


def text_cleaning(text):
    '''Function to clean the text'''
    # convert to lower case
    text = text.lower()
    # remove the email address
    text = text.replace(r'[a-zA-z0-9._]+@[a-zA-z0-9._]+', 'emailaddr')
    # replace the URL's
    text = text.replace(r'(http[s]?\S+)|(\w+\.[a-zA-Z]{2,4}\S*)', 'httpaddr')
    # replace currency symbol with moneysymb
    text = text.replace(r'£|\$', 'moneysymb')
    # Replace phone numbers with 'phonenumbr'
    text = text.replace(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr')
    # remove the ip address
    text = text.replace(r'((2[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '')
    # remove the user handles
    text = text.replace(r'@\w+', '')
    # remove the string punctuation
    text = ' '.join([txt for txt in word_tokenize(text) if txt not in string.punctuation])
    # replace the digits from the text with numbr
    text = text.replace(r'\d', 'numbr')
    # remove all non alphabetical characters
    text = ' '.join([txt for txt in word_tokenize(text) if txt.isalpha()])
    # replace multiple white space with a single one
    text = text.replace(r'\s+', ' ')
    # remove the leading and trailing whitespaces
    text = text.replace(r'^\s+|\s+?$', '')
    # remove the stop words
    text = ' '.join([txt for txt in word_tokenize(text) if txt not in stopwords.words('english')])
    # apply stemmer
    text = ' '.join([stemmer.stem(txt) for txt in word_tokenize(text)])

    # return the cleaned text
    return text


def make_tidy(sample_space, train_scores, valid_scores):
    # Join train_scores and valid_scores, and label with sample_space
    messy_format = pd.DataFrame(np.stack((sample_space, train_scores.mean(axis=1), valid_scores.mean(axis=1)), axis=1),
                                columns=['# of training examples', 'Training set', 'Validation set'])
    # Re-structure into into tidy format
    return pd.melt(messy_format, id_vars='# of training examples', value_vars=['Training set', 'Validation set'],
                   var_name='Scores', value_name='F1 score')


def test_mail_is_spam_or_not(mailBody):
    # load the spam collection mail file
    path = settings.MEDIA_ROOT + "\\" + 'mailSpamCollection.csv'
    sms = pd.read_table(path, names=['label', 'message'])
    sms.head()
    # sms.info()
    sms['label'].value_counts()
    # countplot of the categories of sentiments
    # sns.countplot(sms['label'])
    # plt.title('Countplot of the MAIL labels (ham or spam)')
    # plt.show()
    sms.isna().sum()
    # check for any empty string
    blanks = []
    for idx, rev, lab in sms.itertuples():
        if type(rev) == 'str':
            if rev.isspace():
                blanks.append(idx)
    # print(blanks)
    X = sms.drop('label', axis=1)
    y = sms['label']
    test_size = 0.3
    random_state = 42
    # split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=sms['label'])
    X_train.shape, X_test.shape
    stemmer = PorterStemmer()

    '''
    Clean the text values in train and test
    '''
    preprocesses = [simplify, text_cleaning]

    for preprocess in preprocesses:
        X_train['message'] = X_train['message'].apply(preprocess)
        X_test['message'] = X_test['message'].apply(preprocess)
    # Construct a design matrix using an n-gram model and a tf-idf statistics
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    counts = vectorizer.fit_transform(X_train['message'])
    vocab = vectorizer.vocabulary_
    test_counts = vectorizer.transform(X_test['message'])
    # label encode the target features
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    # Train SVM with a linear kernel on the training set
    clf = LinearSVC(loss='hinge')
    clf.fit(counts, y_train)
    train_predictions = clf.predict(counts)
    test_predictions = clf.predict(test_counts)
    # print('Accuracy Score on the Train and Test Set')
    # print('Train Accuracy = {}'.format(accuracy_score(y_train, train_predictions)))
    # print('Test Accuracy = {}'.format(accuracy_score(y_test, test_predictions)))

    # print('Classification Report on Test Set')
    # print(classification_report(y_test, test_predictions))
    # print('F-1 score on the test set')
    # print('Test Set F1 Score = {}'.format(f1_score(y_test, test_predictions)))

    # print('Confusion Matrix on the Test Set')
    # print(confusion_matrix(y_test, test_predictions))
    # Display a confusion matrix

    pd.DataFrame(confusion_matrix(y_test, test_predictions), index=[['actual', 'actual'], ['spam', 'ham']],
                 columns=[['predicted', 'predicted'], ['spam', 'ham']])
    # select 10 different sizes of the entire training dataset. The test set will still be kept separate an an unseen data
    raw_text = X_train['message']
    sample_space = np.linspace(500, len(raw_text) * .80, 10, dtype='int')

    # Compute learning curves without regularization for the SVM model
    train_size, train_scores, valid_scores = learning_curve(estimator=LinearSVC(loss='hinge', C=1e10),
                                                            X=counts, y=y_train,
                                                            shuffle=True,
                                                            train_sizes=sample_space,
                                                            cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2,
                                                                                      random_state=42),
                                                            scoring='f1',
                                                            n_jobs=-1)
    train_size
    # check the train scores
    train_scores
    # checl the test scores
    valid_scores

    # Initialize a FacetGrid object using the table of scores and facet on the type of score
    g = sns.FacetGrid(make_tidy(sample_space, train_scores, valid_scores), hue='Scores', size=5)
    # Plot the learning curves and add a legend
    # g.map(plt.scatter, '# of training examples', 'F1 score')
    # g.map(plt.plot, '# of training examples', 'F1 score').add_legend()
    # plt.show()
    ''' Grid Search for the best hyperparameter '''
    space = dict()
    space['penalty'] = ['l1', 'l2', 'elasticnet']
    space['loss'] = ['squared_hinge', 'hinge']
    space['C'] = [1e10, 100]
    # print(space)

    clf = LinearSVC()
    folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(estimator=clf, param_grid=space, scoring='f1',
                               n_jobs=-1, cv=folds)
    grid_result = grid_search.fit(counts, y_train)

    grid_result.best_params_
    # clf = LinearSVC(loss='hinge',penalty='l2',C=100)
    clf = grid_result.best_estimator_
    clf.fit(counts, y_train)
    train_predictions = clf.predict(counts)
    test_predictions = clf.predict(test_counts)

    # print('Accuracy Score on the Train and Test Set')
    # print('Train Accuracy = {}'.format(accuracy_score(y_train, train_predictions)))
    # print('Test Accuracy = {}'.format(accuracy_score(y_test, test_predictions)))
    # print('Classification Report on Test Set')
    # print(classification_report(y_test, test_predictions))
    # Display a confusion matrix

    pd.DataFrame(confusion_matrix(y_test, test_predictions), index=[['actual', 'actual'], ['spam', 'ham']],
                 columns=[['predicted', 'predicted'], ['spam', 'ham']])

    # Display the features with the highest weights in the SVM model
    pd.Series(clf.coef_.T.ravel(), index=vectorizer.get_feature_names()).sort_values(ascending=False)[:20]
    # function to decide whether a string is spam or not, and apply it on the hypothetical message from earlier.

    mail = 'Ohhh, but those are the best kind of foods'
    rs = ''
    text = simplify(mailBody)
    text_clean = text_cleaning(text)
    if clf.predict(vectorizer.transform([text_clean])):
        rs = 'spam'
    else:
        rs = 'not spam'
    # print("Spam Results:", rs)
    return rs
