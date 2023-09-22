import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from django.conf import settings

sns.set_style('whitegrid')
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer


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


def start_cleaning_process():
    path = settings.MEDIA_ROOT + "\\" + 'All_Emails.xlsx'
    df = pd.read_excel(path)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.columns = ['Label', 'Text', 'Label_Number']
    # print(df.head())
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Label')
    # plt.show()
    df['count'] = df['Text'].apply(count_words)
    print(df['count'])
    df.groupby('Label_Number')['count'].mean()
    print('Before cleaning:')
    print(df.head())
    print('After cleaning:')
    df['Text'] = df['Text'].apply(lambda string: clean_str(string))
    return df
