import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder as LE
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split as tts



stop_words = set(stopwords.words('english'))

def cleanup(sentence):
    word_tok = nltk.word_tokenize(sentence)
    stemmed_words = [w for w in word_tok if not w in stop_words]
    return ' '.join(stemmed_words)

le = LE()
tfv = TfidfVectorizer(min_df=1, stop_words='english')
data = pd.read_csv(r"C:/Users/ADMIN/Downloads/project/BankFAQs.csv")
questions = data['Question'].values
X = []
for question in questions:
    X.append(cleanup(question))
tfv.fit(X)
le.fit(data['Class'])
X = tfv.transform(X)
y = le.transform(data['Class'])
trainx, testx, trainy, testy = tts(X, y, test_size=.3, random_state=42)
model = SVC(kernel='linear')
model.fit(trainx, trainy)
class_ = le.inverse_transform(model.predict(X))

def get_max5(arr):
    ixarr = []
    for ix, el in enumerate(arr):
        ixarr.append((el, ix))
    ixarr.sort()
    ixs = []
    for i in ixarr[-5:]:
        ixs.append(i[1])
    return ixs[::-1]

def get_response(usrText):
    while True:
        if usrText.lower() == "bye":
            return "Bye"
        GREETING_INPUTS = ["hello", "hi", "greetings", "sup", "what's up", "hey", "hiii", "hii", "yo"]
        a = [x.lower() for x in GREETING_INPUTS]
        sd = ["Thanks", "Welcome"]
        d = [x.lower() for x in sd]
        am = ["OK"]
        c = [x.lower() for x in am]
        t_usr = tfv.transform([cleanup(usrText.strip().lower())])
        class_ = le.inverse_transform(model.predict(t_usr))
        questionset = data[data['Class'].values == class_]
        cos_sims = []
        for question in questionset['Question']:
            sims = cosine_similarity(tfv.transform([question]), t_usr)
            cos_sims.append(sims)
        ind = cos_sims.index(max(cos_sims))
        b = [questionset.index[ind]]
        if usrText.lower() in a:
            return ("Hi, I'm Chatbot, how can i help you!\U0001F60A")
        if usrText.lower() in c:
            return "Ok...Alright!\U0001F64C"
        if usrText.lower() in d:
            return ("My pleasure! \U0001F607")
        if max(cos_sims) > [[0.]]:
            a = data['Answer'][questionset.index[ind]]+"   "
            return a
        elif max(cos_sims)==[[0.]]:
            return "sorry! \U0001F605"

def get_suggestions():
    data = pd.read_csv(r'C:/Users/ADMIN/Downloads/project/venv/BankFAQs.csv')
    return list(data['Question'].str.capitalize())

import requests
import streamlit as st
st.title('SBI CHATBOT')
st.write('**Hi, how can I help you?**')

usrText=st.text_input('Enter your Question here ...')

if usrText:
    st.write('Answer :')
    with st.spinner('Searching for answer.....'):
        prediction=get_response(usrText)
        st.write('answer: {}'.format(prediction))
        #st.write('title: {}'.format(prediction[1]))
        #st.write('paragraph: {}'.format(prediction[2]))
    st.write("")
    

