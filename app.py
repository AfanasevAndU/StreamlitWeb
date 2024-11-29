import streamlit as st
import pickle

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return "Positive" if prediction[0] == 1 else "Negative"

st.title("Анализ тональности текста")

user_input = st.text_area("Введите текст для анализа:", "")

if st.button("Анализировать"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.write(f"**Тональность текста:** {sentiment}")
    else:
        st.warning("Пожалуйста, введите текст.")

st.sidebar.header("Информация о модели")
st.sidebar.write("Модель обучена на IMDB Movie Reviews Dataset.")
st.sidebar.write(f"Алгоритм: Logistic Regression")