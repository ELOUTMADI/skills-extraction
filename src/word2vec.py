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
import os

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

# Function to plot results and save images
def plot_results(results_df, metric, save_dir='plots'):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for param in param_grid:
        plt.figure(figsize=(10, 6))
        for value in results_df[param].unique():
            subset = results_df[results_df[param] == value]
            plt.plot(subset.index, subset[metric], marker='o', label=f'{param}={value}')
        plt.title(f'Effect of {param} on {metric}')
        plt.xlabel('Experiment Index')
        plt.ylabel(metric)
        plt.legend()
        
        # Save the plot
        filename = f'{save_dir}/{param}_{metric}.png'
        plt.savefig(filename)
        plt.close()

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


# Parameter grid
param_grid = {
    'vector_size': [50, 100, 200],
    'window': [3, 5, 10],
    'min_count': [1, 2, 5],
    'sg': [0, 1],  # 0 for CBOW, 1 for Skip-gram
    'hs': [0, 1],  # 0 for negative sampling, 1 for hierarchical softmax
    'negative': [5, 10, 15],  # Number of negative samples if hs=0
    'epochs': [10, 20, 30]  # Number of training epochs
}

# Store results
results = []

# Train and evaluate models
for vector_size in param_grid['vector_size']:
    for window in param_grid['window']:
        for min_count in param_grid['min_count']:
            for sg in param_grid['sg']:
                for hs in param_grid['hs']:
                    for negative in param_grid['negative']:
                        for epochs in param_grid['epochs']:
                            if hs == 1 and negative > 0:
                                continue  # Skip incompatible combinations
                            model = Word2Vec(
                                df['tokenized_description'],
                                vector_size=vector_size,
                                window=window,
                                min_count=min_count,
                                sg=sg,
                                hs=hs,
                                negative=negative,
                                epochs=epochs,
                                workers=4
                            )
                            
                            # Convert descriptions to average word vectors
                            df['average_word_vector'] = df['tokenized_description'].apply(lambda x: get_average_word_vector(x, model, vector_size))
                            X = np.vstack(df['average_word_vector'].values)
                            y = df['Industry']
                            
                            # Split data into train and test sets
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            
                            # Train a classifier
                            clf = LogisticRegression(max_iter=1000)
                            clf.fit(X_train, y_train)
                            
                            # Predict and evaluate
                            y_pred = clf.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            
                            results.append({
                                'vector_size': vector_size,
                                'window': window,
                                'min_count': min_count,
                                'sg': sg,
                                'hs': hs,
                                'negative': negative,
                                'epochs': epochs,
                                'accuracy': accuracy,
                                'f1_score': f1
                            })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plot results for accuracy and F1 score
plot_results(results_df, 'accuracy')
plot_results(results_df, 'f1_score')