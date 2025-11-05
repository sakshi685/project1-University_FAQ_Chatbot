import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity

# --- Load data and model ---
# In a real Streamlit app, you would load these at the beginning of the script.
# For this example, we'll assume they are available globally from the Colab session.
# You might need to adjust paths or loading mechanisms if running this outside Colab.

# Ensure NLTK resources are downloaded (if not already)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt_tab')
except nltk.downloader.DownloadError:
     nltk.download('punkt_tab')


# Assuming 'df' DataFrame is available from the Colab environment
# Assuming 'tokenizer' and 'model' (BERT) are available from the Colab environment


# --- Chatbot Logic Functions ---

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def preprocess_query(query):
    query = query.lower()
    query = re.sub(r'[^a-z\s]', '', query)
    tokens = nltk.word_tokenize(query)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def get_bert_embeddings(texts, tokenizer, model):
    # Handle single text input
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the embeddings of the [CLS] token as sentence representation
    return outputs.last_hidden_state[:, 0, :].numpy()

def find_best_answer_bert(user_query_embedding, faq_embeddings, df):
    similarity_scores = cosine_similarity(user_query_embedding.reshape(1, -1), faq_embeddings)
    best_match_index = similarity_scores.argmax()
    best_answer = df.loc[best_match_index, 'Answer']
    return best_answer

# --- Streamlit App Interface ---

st.title("University FAQ Chatbot")

st.write("Ask me a question about the university. Type 'quit' to exit (in the Colab console if running there).")

user_input = st.text_input("Your question:")

if user_input:
    # Assuming df, tokenizer, and model are loaded and available in the Colab environment
    # In a standalone Streamlit app, you would load these here.
    # For demonstration in Colab, we rely on the Colab environment's state.
    try:
        # Access variables from the Colab environment
        global df, tokenizer, model

        if 'df' not in globals() or 'tokenizer' not in globals() or 'model' not in globals():
            st.error("Required variables (df, tokenizer, model) not found. Please run the previous cells to load them.")
        else:
            preprocessed_user_query = preprocess_query(user_input)
            user_query_embedding = get_bert_embeddings(preprocessed_user_query, tokenizer, model)
            # Assuming BERT_Embeddings column is already created in df
            if 'BERT_Embeddings' not in df.columns:
                 st.error("BERT_Embeddings column not found in DataFrame. Please run the embedding generation cell.")
            else:
                faq_embeddings = np.array(df['BERT_Embeddings'].tolist())
                best_answer = find_best_answer_bert(user_query_embedding, faq_embeddings, df)
                st.write("Chatbot:", best_answer)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        #st.write("Please make sure the FAQ data and BERT model are loaded correctly.")
