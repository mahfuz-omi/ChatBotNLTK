import nltk
import pandas as pd
import numpy as np

pd.set_option('display.expand_frame_repr', False)

df = pd.read_excel("dialog_talk_agent.xlsx")

# fill row to row .so axix=0
# if axix = 1, then go column to column
df.ffill(inplace=True,axis=0)


def normalizeText(text):
    return str(text).lower()


df['question'] = df['Context'].apply(normalizeText)
df['answer'] = df['Text Response'].apply(normalizeText)

df.drop(['Context', 'Text Response'], axis=1,inplace = True)

from sklearn.feature_extraction import text
# countvector only takes 1d array
# need to provide the whole train-test data to generate vector that can hold any text
# this fit determines the length of each vector
# length of vectors must be equal for train and test and any new data
# if the new data has the
vectorizer = text.CountVectorizer(binary=False)
vectorizer.fit(df['question'].values)

sample_question = str(input('Input Question( C to exit:  '))
while sample_question != "c":

    sample_question_vector = vectorizer.transform([sample_question])


    from sklearn.metrics.pairwise import cosine_similarity

    # put cosine score in each row
    def cosine_put(x):
        vector_row = vectorizer.transform([x])
        cosine_score = cosine_similarity(sample_question_vector,vector_row)
        return cosine_score[0][0]

    df['cosine_score'] = df.apply(lambda row:cosine_put(row['question']),axis=1)


    # sort w.r.t cosine score
    df.sort_values("cosine_score",ascending=False,inplace=True)
    # print first result answer
    print(df['answer'].values[0])

    sample_question = str(input('Input Question( C to exit:  '))











