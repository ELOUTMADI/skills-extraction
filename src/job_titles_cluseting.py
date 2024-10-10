import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from deep_translator import GoogleTranslator

from transformers import BertTokenizer, BertModel
import torch

# Ensure nltk resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize the translator
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Initialize the stopwords and lemmatizer
stop_words = set(stopwords.words('english') + ["junior", "senior", "assistant", "f/h", "h/f","f/m","m/f" "expérimenté(e)","experte","expérimentée","experienced",""])
lemmatizer = WordNetLemmatizer()

# Function to translate job titles from French to English

def translate_to_english(titles, beam_size=1):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer.src_lang = "fr_XX"

    encoded_fr = tokenizer(titles, return_tensors="pt", padding=True).to(device)
    generated_tokens = model.generate(
        **encoded_fr,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
        num_beams=beam_size,  # Lower beam size for faster translation
        max_length=512,  # Adjust max_length if your texts are long
        no_repeat_ngram_size=3,  # Prevents repetition
    )
    translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translations[0]

# Function to clean and preprocess job titles
def preprocess_job_title(title):
    # Convert to lowercase
    title = title.lower()
    
    # Remove specific characters like "/", "-", and others
    title = re.sub(r'[\/\-()]', ' ', title)
    
    # Tokenize the title
    words = title.split()
    
    # Remove stopwords and apply lemmatization/stemming
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Join the cleaned words back into a string
    cleaned_title = ' '.join(cleaned_words)
    
    return cleaned_title

# Load the dataset
df = pd.read_csv('src/job_titles2.csv')

# Apply translation and preprocessing to job titles
df['Title'] = df['Title'].apply(translate_to_english)
df['Title'] = df['Title'].apply(preprocess_job_title)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get embeddings for a title
def get_bert_embedding(title):
    inputs = tokenizer(title, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    # Use the mean of the token embeddings as the representation for the title
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings.flatten()

# Generate embeddings for all job titles
embeddings = df['Title'].apply(get_bert_embedding).tolist()
embeddings = torch.tensor(embeddings)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Convert embeddings to numpy array for clustering
embeddings_np = embeddings.numpy()

# Determine the optimal number of clusters using the Elbow method or Silhouette score
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings_np)
    score = silhouette_score(embeddings_np, labels)
    silhouette_scores.append(score)

# Optimal cluster count (You can select based on the highest silhouette score or an elbow plot)
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(embeddings_np)

# Visualize the results using PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce dimensions for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings_np)

# Plot the clusters
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=df['Cluster'], cmap='viridis')
plt.title('K-Means Clustering of Job Titles')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Function to create a word cloud for a specific cluster
def plot_word_cloud(cluster_num, titles):
    # Combine all titles into a single string
    combined_text = ' '.join(titles)
    
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(combined_text)
    
    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Cluster {cluster_num}')
    plt.show()

# Extract job titles for each cluster and plot word clouds
for cluster_num in df['Cluster'].unique():
    cluster_titles = df[df['Cluster'] == cluster_num]['Title'].tolist()
    plot_word_cloud(cluster_num, cluster_titles)
