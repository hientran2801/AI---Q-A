import pandas as pd
import numpy as np
import string

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df_08 = pd.read_csv('input/questionanswer-dataset/S08_question_answer_pairs.txt', sep='\t')
df_09 = pd.read_csv('input/questionanswer-dataset/S09_question_answer_pairs.txt', sep='\t')
df_10 = pd.read_csv('input/questionanswer-dataset/S10_question_answer_pairs.txt', sep='\t', encoding = 'ISO-8859-1')
print(df_08)
df_all = df_08.append([df_09, df_10])

df_all_1 = df_all[['Question', 'Answer']]
df_all_1.head()

df_all_1 =df_all_1.dropna(axis=0)
df_all_2 = df_all_1.drop_duplicates(subset='Question')


def getResults(questions, func):
    def getResult(q):
        answer, score, prediction = func(q)
        return [q, prediction, answer, score]
    return pd.DataFrame(list(map(getResult, questions)), columns=["Question", "Prediction", "Answer", "Score"])
test_data = [
    "Why you don't love me?",
    "At what age can a zebra breed?",
    "Did his mother die of pneumonia?",
    "Do you know the length of leopard's tail?",
    "When polar bears can be invisible?",
    "Can I see arctic animals?",
    "some city in Finland",
    "zebra eat"
]

from Levenshtein import ratio
def getApproximateAnswer(q):
    max_score = 0
    answer = ""
    prediction = ""
    for idx, row in df_all_2.iterrows():
        score = ratio(row["Question"], q)
        if score >= 0.9:
            return row["Answer"], score, row["Question"]
        elif score > max_score:
            max_score = score
            answer = row["Answer"]
            prediction = row["Question"]
    if max_score > 0.3:
        print(answer, max_score, prediction)
        return answer, max_score, prediction
    return "Sorry, I didn't get you.", max_score, prediction
print(getResults(test_data, getApproximateAnswer))