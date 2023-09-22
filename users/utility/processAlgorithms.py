import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from django.conf import settings

sns.set_style('whitegrid')

from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import accuracy_score, classification_report,precision_score,recall_score


def count_words(text):
    words = word_tokenize(text)
    return len(words)


# %%time
def clean_str(string, reg=RegexpTokenizer(r'[a-z]+')):
    # Clean a string with RegexpTokenizer
    string = string.lower()
    tokens = reg.tokenize(string)
    return " ".join(tokens)


from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def stemming(text):
    return ''.join([stemmer.stem(word) for word in text])


def start_process():
    path = settings.MEDIA_ROOT + "\\" + 'All_Emails.xlsx'
    df = pd.read_excel(path)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.columns = ['Label', 'Text', 'Label_Number']
    # print(df.head())
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Label')
    plt.show()
    df['count'] = df['Text'].apply(count_words)
    print(df['count'])
    df.groupby('Label_Number')['count'].mean()
    print('Before cleaning:')
    print(df.head())
    print('After cleaning:')
    df['Text'] = df['Text'].apply(lambda string: clean_str(string))
    print(df.head())
    df['Text'] = df['Text'].apply(stemming)
    print(df.head())
    X = df.loc[:, 'Text']
    y = df.loc[:, 'Label_Number']

    print(f"Shape of X: {X.shape}\nshape of y: {y.shape}")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11)
    print(f"Training Data Shape: {X_train.shape}\nTest Data Shape: {X_test.shape}")
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    cv.fit(X_train)
    print('No.of Tokens: ', len(cv.vocabulary_.keys()))
    dtv = cv.transform(X_train)
    type(dtv)
    dtv = dtv.toarray()
    print(f"Number of Observations: {dtv.shape[0]}\nTokens/Features: {dtv.shape[1]}")
    dtv[1]
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import LinearSVC, SVC
    from time import perf_counter
    import warnings
    warnings.filterwarnings(action='ignore')
    models = {
        "Random Forest": {"model": RandomForestClassifier(), "perf": 0},
        "MultinomialNB": {"model": MultinomialNB(), "perf": 0},
        "Logistic Regr.": {"model": LogisticRegression(solver='liblinear', penalty='l2', C=1.0), "perf": 0},
        "KNN": {"model": KNeighborsClassifier(), "perf": 0},
        "Decision Tree": {"model": DecisionTreeClassifier(), "perf": 0},
        "SVM (Linear)": {"model": LinearSVC(), "perf": 0},
        "SVM (RBF)": {"model": SVC(), "perf": 0}
    }

    for name, model in models.items():
        start = perf_counter()
        model['model'].fit(dtv, y_train)
        duration = perf_counter() - start
        duration = round(duration, 2)
        model["perf"] = duration
        print(f"{name:20} trained in {duration} sec")
    test_dtv = cv.transform(X_test)
    test_dtv = test_dtv.toarray()
    print(f"Number of Observations: {test_dtv.shape[0]}\nTokens: {test_dtv.shape[1]}")
    models_accuracy = []
    for name, model in models.items():
        models_accuracy.append([name, model["model"].score(test_dtv, y_test), model["perf"]])
    df_accuracy = pd.DataFrame(models_accuracy)
    df_accuracy.columns = ['Model', 'Test Accuracy', 'Training time (sec)']
    df_accuracy.sort_values(by='Test Accuracy', ascending=False, inplace=True)
    df_accuracy.reset_index(drop=True, inplace=True)
    print(df_accuracy)

    # plt.figure(figsize=(15, 5))
    # sns.barplot(x='Model', y='Test Accuracy', data=df_accuracy)
    # plt.title('Accuracy on the test set\n', fontsize=15)
    # plt.ylim(0.825, 1)
    # plt.show()

    # plt.figure(figsize=(15, 5))
    # sns.barplot(x='Model', y='Training time (sec)', data=df_accuracy)
    # plt.title('Training time for each model in sec', fontsize=15)
    # plt.ylim(0, 1)
    # plt.show()

    lr = LogisticRegression(solver='liblinear', penalty='l2', C=1.0)
    lr.fit(dtv, y_train)
    pred = lr.predict(test_dtv)
    print('Accuracy: ', accuracy_score(y_test, pred) * 100)
    print(classification_report(y_test, pred))
    # confusion_matrix = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
    # plt.figure(figsize=(6, 6))
    # sns.heatmap(confusion_matrix, annot=True, cmap='Paired', cbar=False, fmt="d", xticklabels=['Not Spam', 'Spam'],
    #             yticklabels=['Not Spam', 'Spam'])
    #  plt.show()

    # Decision Tree Classifier
    dtc = DecisionTreeClassifier()
    dtc.fit(dtv, y_train)
    pred = dtc.predict(test_dtv)
    accuracy =  accuracy_score(y_test, pred) * 100
    precision = precision_score(y_test, pred) * 100
    recall = recall_score(y_test,pred) * 100
    print(classification_report(y_test, pred))
    confusion_matrix = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix, annot=True, cmap='Paired', cbar=False, fmt="d", xticklabels=['Not Spam', 'Spam'],
                yticklabels=['Not Spam', 'Spam']);
    plt.show()

    return accuracy,precision,recall
