import pytest
import pickle

# Загрузка модели и векторизатора
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Функция для тестирования предсказания
def test_predict_sentiment_positive():
    text = "The movie was absolutely wonderful! I loved it."
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    assert prediction[0] == 1, "Positive text was not classified correctly"

def test_predict_sentiment_negative():
    text = "The movie was terrible. I hated it."
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    assert prediction[0] == 0, "Negative text was not classified correctly"

def test_model_loaded():
    assert model is not None, "Model is not loaded"
    assert vectorizer is not None, "Vectorizer is not loaded"
