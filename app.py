import streamlit as st
import pickle

# Загрузка модели и векторизатора
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Функция для анализа тональности
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return "Positive" if prediction[0] == 1 else "Negative"

# Интерфейс Streamlit
st.title("Анализ тональности текста")

# Ввод текста
user_input = st.text_area("Введите текст для анализа:", "")

if st.button("Анализировать"):
    if user_input.strip():
        sentiment = predict_sentiment(user_input)
        st.write(f"**Тональность текста:** {sentiment}")
    else:
        st.warning("Пожалуйста, введите текст.")

# Информация о модели
st.sidebar.header("Информация о модели")
st.sidebar.write("Модель обучена на IMDB Movie Reviews Dataset.")
st.sidebar.write(f"Алгоритм: Logistic Regression")