import numpy as np
import pandas as pd 
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
import re
import matplotlib.pyplot as plt
from textblob import TextBlob, Word
import nltk
from gensim.models.phrases import Phraser
from gensim.models import Phrases
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Load the data 
ds = pd.read_csv('job_Data_Scientist_VietNam.csv')
ds['Industry'] = 'Data Scientist'

mb = pd.read_csv('job_Mobile_Developer_Vit_Nam.csv')
mb['Industry'] = 'Mobile Developer'

df = pd.concat([ds,mb], ignore_index=True)

df = df[['title', 'company', 'location', 'description', 'Industry']]
df = df.dropna().reset_index(drop=True)

# Detect language
from langdetect import detect
import re
from gensim.models import Word2Vec
def detect_lang(text):
    try:
        return detect(text)
    except:
        return 'unknown'
    
df['lang'] = df['description'].apply(detect_lang)

# Filter out non-English job descriptions
df = df[df['lang'] == 'en']

# Drop Duplicates
df = df.drop_duplicates(subset=df.columns.difference(['Industry'])).reset_index(drop=True)


# Function to preprocess text
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Remove digits
    text = ''.join([i for i in text if not i.isdigit()])
    # Lemmatization
    text = ' '.join([Word(word).lemmatize() for word in text.split()])
    return text


# Count number of line in each job description to check if the data is balanced
df['line_count'] = df['description'].apply(lambda x: len(x.split('\n')))


# Visualize Word Count function for the job description

def visualize_word_count(df):
    # Concatenate all job descriptions into a single string
    all_descriptions = ' '.join(df['description'])
    
    # Preprocess the text
    preprocessed_text = preprocess_text(all_descriptions)
    
    # Create a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(preprocessed_text)
    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def remove_stop_words(df):
    ## Delete more stop words
    other_stop_words = ['intern', 'junior', 'senior','experience','etc','job','work','company','technique',
                        'candidate','language','menu','inc','new','plus','years',
                       'technology','organization','ceo','cto','account','manager','scientist','mobile',
                        'developer','product','revenue','strong', 'work', 'team', 'include', 'well', 'join_us',
                        'excellent', 'belong', 'hybrid', 'working', 'enable_company',
                        'excellent_opportunity_advancement']


    # Join stop words with '|', creating a regex pattern
    stop_words_pattern = '|'.join(r'\b{}\b'.format(word) for word in other_stop_words)

    # Apply regex substitution to remove stop words from 'description'
    df['description'] = df['description'].apply(lambda x: re.sub(stop_words_pattern, '', x, flags=re.IGNORECASE))
    return df

df = remove_stop_words(df)

#visualize_word_count(df)


print("start2")


# Tokenize the description
df['description'] = df['description'].apply(preprocess_text)
df['tokenized_description'] = df['description'].apply(lambda x: word_tokenize(x.lower()))



# Detect bigrams and trigrams
#phrases = Phrases(df['tokenized_description'], min_count=5, threshold=100)
#bigram = Phraser(phrases)
#df['tokenized_description'] = df['tokenized_description'].apply(lambda x: bigram[x])

# Train the Word2Vec model
model = Word2Vec(df['tokenized_description'], vector_size=200, window=10, min_count=1, workers=4,sg=1,negative=15, epochs=10)


# Function to get the average word vector for a job description
def get_average_word_vector(description, model, vector_size):
    word_vectors = np.array([model.wv[word] for word in description if word in model.wv.key_to_index])    
    if len(word_vectors) == 0:
        return np.zeros(vector_size)
    
    average_vector = np.mean(word_vectors, axis=0)
    
    return average_vector

# Get the average word vector for each job description
df['average_word_vector'] = df['tokenized_description'].apply(lambda x: get_average_word_vector(x, model, 200))


# Find Similar words to Python
N = 20

technical_skills = ['python']
for word in technical_skills:
    try:
        similar_word = model.wv.most_similar(word, topn=N)
        print("Similar words to <<",word,">>", similar_word, '\n')
    except:
        print("No", word, "available \n")


