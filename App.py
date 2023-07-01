import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

# required function
def text_transform(text):
    rtext = text.lower()  # lower case

    rtext = nltk.word_tokenize(rtext)  # tokenize the words

    y = []
    for i in rtext:
        if i.isalnum():
            y.append(i)

    rtext = y[:]
    y.clear()

    for i in rtext:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    rtext = y[:]
    y.clear()

    # stemming
    for i in rtext:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open("Vectorizer.pkl", "rb"))
model = pickle.load(open("Model.pkl", "rb"))

st.title("Email Spam Classifier")
st.divider()
st.header("About the App")
st.write(
    "This The Email Spam Classifier is an intelligent system designed to accurately identify and classify spam emails. It utilizes advanced machine learning algorithms and natural language processing techniques to analyze the content, structure, and metadata of incoming emails, enabling efficient spam detection and filtering"
)
st.write("Algorithm used is : Multinomial Naive Bayes with TF-IDF vectorizer")
st.divider()

st.subheader("Predict if your message is spam or not")
input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    # Preprocess
    transformed_sms = text_transform(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # predict
    result = model.predict(vector_input)[0]

    # Display
    if result == 1:
        st.divider()
        st.header("Spam")
    else:
        st.divider()
        st.header("Not Spam")
