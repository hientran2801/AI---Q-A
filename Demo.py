import pandas as pd
import pymongo
import numpy as np
import string
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from vncorenlp import VnCoreNLP
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel


def get_stopwords_list(stop_file_path):
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return list(frozenset(stop_set))

stopwords_path = 'input/vietnamese-stopwords'
stopwords_list = get_stopwords_list(stopwords_path)

myclient = pymongo.MongoClient('mongodb://localhost:27017/')
mydb = myclient['mydatabase']
mycol = mydb["database"]


cursor = mycol.find()
all_data = pd.DataFrame(list(cursor))
all_data['AnswerA'] = all_data['AnswerA'] + '\n' + all_data['AnswerB'] + '\n' + all_data['AnswerC'] + '\n' + all_data['AnswerD']
all_data = all_data[['Question', 'Answer', 'AnswerA']]
all_data.dropna(axis=0)
all_data.drop_duplicates(subset='Question')


lemmatizer = WordNetLemmatizer()
annotator = VnCoreNLP("<FULL-PATH-to-VnCoreNLP-jar-file>", annotators="wseg,pos", max_heap_size='-Xmx2g')

def my_tokenizer(doc):
    # words = word_tokenize(doc)
    # print(words)
    # pos_tags = pos_tag(words)
    # print(pos_tags)
    pos_tags = annotator.annotate(doc)
    non_stopwords = [w for w in pos_tags if not w[0].lower() in stopwords_list]
    non_punctuation = [w for w in non_stopwords if not w[0] in string.punctuation]

    lemmas = []
    for w in non_punctuation:
        if w[1].startswith('J'):
            pos = wordnet.ADJ
        elif w[1].startswith('V'):
            pos = wordnet.VERB
        elif w[1].startswith('N'):
            pos = wordnet.NOUN
        elif w[1].startswith('R'):
            pos = wordnet.ADV
        else:
            pos = wordnet.NOUN

        lemmas.append(lemmatizer.lemmatize(w[0], pos))
    return lemmas


# tfidf_vectorizer = TfidfVectorizer(tokenizer=my_tokenizer)
# tfidf_matrix = tfidf_vectorizer.fit_transform(tuple(all_data['Question']))
# pickle.dump(tfidf_vectorizer, open("crawlData/myvectorizer.pickle", "wb"))
# sparse.save_npz("crawlData/mymatrix.npz", tfidf_matrix)

tfidf_vectorizer = pickle.load(open("crawlData/myvectorizer.pickle", "rb"))
tfidf_matrix = sparse.load_npz("crawlData/mymatrix.npz")

def ask_question(question):
    query_vect = tfidf_vectorizer.transform([question])
    similarity = cosine_similarity(query_vect, tfidf_matrix)
    # print(query_vect)
    # similarity = SVC(kernel=chi2_kernel).fit(tfidf_matrix, query_vect)
    # similarity.predict
    max_similarity = np.argmax(similarity, axis=None)

    if (np.amax(similarity, axis=None) > 0.4):
        print('Closest question found:', all_data.iloc[max_similarity]['Question'])
        print('Similarity: {:.2%}'.format(similarity[0, max_similarity]))
        print('Option: \n', all_data.iloc[max_similarity]['AnswerA'])
        print('Answer:', all_data.iloc[max_similarity]['Answer'])
    else :
        print("Sorry, I didn't get you.")

question_active = True

while question_active:
    question_user = input('Type your question here: ')
    ask_question(question_user)
    response = input('Do you have any questions? (y/n)')
    if response == 'n':
        question_active = False;
        print("Good bye!")


