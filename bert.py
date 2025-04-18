import scipy.sparse as sp
import datetime
import itertools
import seaborn as sns
import time
import re
import string
import chardet
import html
from collections import Counter
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf

# Caricamento dati
reviews_a = pd.read_csv("amazon_reviews_99-2012.csv", encoding="MacRoman")

# Converti i timestamp in oggetti datetime
reviews_a['Time'] = pd.to_datetime(reviews_a['Time'], unit='s')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Dizionario di slang e acronimi (aggiungere altri se necessario)
slang_dict = {
    "u": "you", "ur": "your", "r": "are", "idk": "I don't know", "btw": "by the way",
    "imo": "in my opinion", "imho": "in my humble opinion", "lol": "laughing out loud",
    "lmao": "laughing my ass off", "brb": "be right back", "gtg": "got to go",
    "smh": "shaking my head", "omg": "oh my god", "thx": "thanks", "ty": "thank you",
    "pls": "please", "plz": "please", "gr8": "great", "b4": "before", "cuz": "because",
    "bc": "because", "wanna": "want to", "gonna": "going to", "gotta": "got to"
}
def clean_text(text):
    # Convertire in minuscolo
    text = text.lower()
    # Decodifica entità HTML (&quot; -> ", &amp; -> &, ecc.)
    text = html.unescape(text)
    # Rimuovere menzioni, URL e hashtag
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Menzioni
    text = re.sub(r'https?://\S+', '', text)   # URL
    text = re.sub(r'#', '', text)               # Rimuove hashtag mantenendo la parola
    text = re.sub(r'www\.\S+', '', text)    # Rimuove URL che iniziano con www

    # Sostituzione di acronimi e slang
    words = text.split()
    words = [slang_dict[word] if word in slang_dict else word for word in words]
    text = ' '.join(words)

    # Rimozione di punteggiatura e numeri
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Tokenizzazione
    tokens = word_tokenize(text)

    # Rimozione stopwords
    stop_words = set(stopwords.words('english'))

    tokens = [word for word in tokens if word not in stop_words]

    # Rimozione della parola "walmart"
    words_to_remove = ["br", "walmart"]
    tokens = [word for word in tokens if word not in words_to_remove]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

# Applicare la pulizia alla colonna "SentimentText"
reviews_a["Text"] = reviews_a["Text"].astype(str).apply(clean_text)
reviews_a["Text"].head
# Bilanciamento delle classi
# Calcola il numero di campioni per ogni classe
class_counts = reviews_a['Score'].value_counts()
print("Conteggio campioni per classe originale:\n", class_counts)
# Identifica la classe con il minor numero di campioni
min_count = class_counts.min()
print("\nNumero minimo di campioni per classe:", min_count)

# Separa il DataFrame per la classe 5 e le altre classi
df_classe_5 = reviews_a[reviews_a['Score'] == 5]
df_classe_4 = reviews_a[reviews_a['Score'] == 4]
df_classe_3 = reviews_a[reviews_a['Score'] == 3]
df_classe_2 = reviews_a[reviews_a['Score'] == 2]
df_classe_1 = reviews_a[reviews_a['Score'] == 1]

# Sottocampiona casualmente le classi al numero di campioni della classe meno rappresentata
df_classe_5_undersampled = df_classe_5.sample(n=min_count, random_state=42) # random_state per la riproducibilità
df_classe_4_undersampled = df_classe_4.sample(n=min_count, random_state=42) # random_state per la riproducibilità
df_classe_3_undersampled = df_classe_3.sample(n=min_count, random_state=42) # random_state per la riproducibilità
df_classe_2_undersampled = df_classe_2.sample(n=min_count, random_state=42) # random_state per la riproducibilità
df_classe_1_undersampled = df_classe_1.sample(n=min_count, random_state=42) # random_state per la riproducibilità

# Combina il DataFrame sottocampionato della classe 5 con le altre classi
reviews_a_balanced = pd.concat([df_classe_1_undersampled, df_classe_2_undersampled, df_classe_3_undersampled, df_classe_4_undersampled, df_classe_5_undersampled])

model_name = 'distilbert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = TFBertModel.from_pretrained(model_name)

# Verifica se una GPU è disponibile per TensorFlow
if tf.config.list_physical_devices('GPU'):
    device = '/GPU:0'
    print("GPU disponibile, utilizzo:", device)
else:
    device = '/CPU:0'
    print("Nessuna GPU disponibile, utilizzo CPU:", device)

# La modalità di valutazione in TensorFlow si gestisce in modo diverso,
# spesso tramite tf.function o semplicemente evitando l'addestramento.

# Funzione per ottenere l'embedding BERT di un testo (adattata per TensorFlow)
def get_bert_embedding(text, tokenizer, model, max_length=128):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors='tf')
    with tf.device(device):
        outputs = model(**encoded_input)
        # Embedding dell'intera sequenza (del token [CLS])
        embedding = outputs.pooler_output.numpy()
        # Alternativa: media degli embedding di tutti i token
        # embedding = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()
    return embedding.flatten()

embedding_dim = bert_model.config.hidden_size

# Identificazione della colonna di testo
if 'Review' in reviews_a_balanced.columns:
    text_column = 'Review'
elif 'Summary' in reviews_a_balanced.columns:
    text_column = 'Summary'
else:
    print("Nessuna colonna 'Review' o 'Summary' trovata. Assicurati che il DataFrame contenga il testo.")
    exit()

print(f"Generazione embedding BERT per la colonna: '{text_column}'")

# Creazione di una lista per contenere tutti gli embedding
all_embeddings = []

# Iterazione attraverso le recensioni e generazione degli embedding
for index, row in reviews_a_balanced.iterrows():
    text = str(row[text_column]) # Assicurati che il testo sia una stringa
    embedding = get_bert_embedding(text, tokenizer, bert_model)
    all_embeddings.append(embedding)

# Conversione della lista di embedding in un array numpy
embeddings_array = np.array(all_embeddings)

# Creazione di nuove colonne nel DataFrame per gli embedding
for i in range(embedding_dim):
    reviews_a_balanced[f'bert_embedding_{i}'] = embeddings_array[:, i]

print(f"Embedding BERT generati e aggiunti come {embedding_dim} nuove colonne al DataFrame 'reviews_a'.")
print(reviews_a_balanced.head())

#proviamo a salvare il dataset reviews_a_balanced
reviews_a_balanced.to_excel(reviews_a_balanced)