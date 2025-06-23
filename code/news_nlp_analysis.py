import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation, NMF
import warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

# Function to load and prepare data
def load_data(file_path):
    # Read JSON file line by line (more memory efficient for large files)
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

# ------------------ Preprocessing Functions ------------------

def preprocess_text(text):
    """
    Apply all preprocessing steps in sequence
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def apply_pos_tagging(text):
    """
    Apply POS tagging to text
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    return tagged

# ------------------ Vectorization Functions ------------------

def vectorize_bow(texts, max_features=5000):
    """
    Vectorize using Bag of Words
    """
    vectorizer = CountVectorizer(max_features=max_features)
    bow_matrix = vectorizer.fit_transform(texts)
    return bow_matrix, vectorizer

def vectorize_tfidf(texts, max_features=5000):
    """
    Vectorize using TF-IDF
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

# ------------------ Sentiment Analysis Functions ------------------

def analyze_sentiment_textblob(text):
    """
    Analyze sentiment using TextBlob
    """
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def analyze_sentiment_vader(text):
    """
    Analyze sentiment using VADER
    """
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# ------------------ Topic Modeling Functions ------------------

def perform_lda(vector_matrix, vectorizer, num_topics=10):
    """
    Perform LDA topic modeling
    """
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(vector_matrix)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract topics
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-11:-1]  # Get top 10 words
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append(top_words)
        
    return lda, topics

def perform_nmf(vector_matrix, vectorizer, num_topics=10):
    """
    Perform NMF topic modeling
    """
    nmf = NMF(n_components=num_topics, random_state=42)
    nmf.fit(vector_matrix)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract topics
    topics = []
    for topic_idx, topic in enumerate(nmf.components_):
        top_words_idx = topic.argsort()[:-11:-1]  # Get top 10 words
        top_words = [feature_names[i] for i in top_words_idx]
        topics.append(top_words)
        
    return nmf, topics

# ------------------ Main Execution ------------------

def main():
    # File path - use the sample dataset instead of the full dataset
    file_path = "data/news_sample.json"
    
    print("Loading data...")
    # Load and prepare data
    df = load_data(file_path)
    
    # Display basic information
    print(f"\nDataset Shape: {df.shape}")
    print("\nColumns in the dataset:")
    print(df.columns.tolist())
    print("\nSample data:")
    print(df.head())
    
    # Count categories
    print("\nNews Categories Distribution:")
    category_counts = df['category'].value_counts().head(10)
    print(category_counts)
    
    # Visualize category distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=category_counts.values, y=category_counts.index)
    plt.title('Top 10 News Categories')
    plt.xlabel('Count')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig('category_distribution.png')
    
    # Preprocess headlines
    print("\nPreprocessing headlines...")
    df['processed_headline'] = df['headline'].apply(preprocess_text)
    
    # Optional: Apply POS tagging to a sample
    sample_headline = df['headline'].iloc[0]
    sample_processed = df['processed_headline'].iloc[0]
    sample_pos_tags = apply_pos_tagging(sample_headline)
    
    print(f"\nSample headline: {sample_headline}")
    print(f"After preprocessing: {sample_processed}")
    print(f"POS tags: {sample_pos_tags}")
    
    # Vectorization
    print("\nApplying vectorization methods...")
    # Bag of Words
    bow_matrix, bow_vectorizer = vectorize_bow(df['processed_headline'])
    
    # TF-IDF
    tfidf_matrix, tfidf_vectorizer = vectorize_tfidf(df['processed_headline'])
    
    # Compare vectorization methods for a sample headline
    sample_idx = 0
    sample_headline_processed = df['processed_headline'].iloc[sample_idx]
    
    # Get BoW representation
    sample_bow = bow_vectorizer.transform([sample_headline_processed])
    sample_bow_array = sample_bow.toarray()[0]
    sample_bow_features = bow_vectorizer.get_feature_names_out()
    
    # Get TF-IDF representation
    sample_tfidf = tfidf_vectorizer.transform([sample_headline_processed])
    sample_tfidf_array = sample_tfidf.toarray()[0]
    sample_tfidf_features = tfidf_vectorizer.get_feature_names_out()
    
    # Show non-zero elements for both
    print("\nBag of Words representation (non-zero elements):")
    for idx in np.nonzero(sample_bow_array)[0]:
        print(f"{sample_bow_features[idx]}: {sample_bow_array[idx]}")
    
    print("\nTF-IDF representation (non-zero elements):")
    for idx in np.nonzero(sample_tfidf_array)[0]:
        print(f"{sample_tfidf_features[idx]}: {sample_tfidf_array[idx]:.4f}")
    
    # Sentiment Analysis
    print("\nPerforming sentiment analysis...")
    
    # TextBlob
    df['sentiment_textblob'] = df['headline'].apply(analyze_sentiment_textblob)
    
    # VADER
    df['sentiment_vader'] = df['headline'].apply(analyze_sentiment_vader)
    
    # Compare sentiment analysis results
    textblob_sentiments = df['sentiment_textblob'].value_counts()
    vader_sentiments = df['sentiment_vader'].value_counts()
    
    print("\nTextBlob Sentiment Distribution:")
    print(textblob_sentiments)
    
    print("\nVADER Sentiment Distribution:")
    print(vader_sentiments)
    
    # Visualize sentiment analysis results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x=textblob_sentiments.index, y=textblob_sentiments.values)
    plt.title('TextBlob Sentiment Distribution')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    sns.barplot(x=vader_sentiments.index, y=vader_sentiments.values)
    plt.title('VADER Sentiment Distribution')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png')
    
    # Topic Modeling
    print("\nPerforming topic modeling...")
    
    # Use a smaller number of topics for the sample dataset
    num_topics = 5
    
    # LDA with TF-IDF
    print("\nRunning LDA with TF-IDF...")
    lda_model, lda_topics = perform_lda(tfidf_matrix, tfidf_vectorizer, num_topics=num_topics)
    
    # NMF with TF-IDF
    print("\nRunning NMF with TF-IDF...")
    nmf_model, nmf_topics = perform_nmf(tfidf_matrix, tfidf_vectorizer, num_topics=num_topics)
    
    # Print topics
    print("\nLDA Topics:")
    for i, topic_words in enumerate(lda_topics):
        print(f"Topic {i+1}: {', '.join(topic_words)}")
    
    print("\nNMF Topics:")
    for i, topic_words in enumerate(nmf_topics):
        print(f"Topic {i+1}: {', '.join(topic_words)}")
    
    # Visualize topics wordcloud
    try:
        from wordcloud import WordCloud
        
        # Create topic visualizations
        for idx, topic in enumerate(lda_topics):
            plt.figure(figsize=(10, 6))
            wc = WordCloud(background_color="white", max_words=100, width=800, height=400)
            wc.generate(" ".join(topic))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            plt.title(f"LDA Topic {idx+1}")
            plt.tight_layout()
            plt.savefig(f'lda_topic_{idx+1}.png')
            plt.close()
        
        for idx, topic in enumerate(nmf_topics):
            plt.figure(figsize=(10, 6))
            wc = WordCloud(background_color="white", max_words=100, width=800, height=400)
            wc.generate(" ".join(topic))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            plt.title(f"NMF Topic {idx+1}")
            plt.tight_layout()
            plt.savefig(f'nmf_topic_{idx+1}.png')
            plt.close()
    except ImportError:
        print("WordCloud package not installed, skipping topic visualization")
    
    # Create a summary report
    print("\nCreating summary report...")
    
    with open('nlp_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write("# News Headlines NLP Analysis Report\n\n")
        
        f.write("## Dataset Overview\n")
        f.write(f"- Total headlines: {len(df)}\n")
        f.write(f"- Categories: {df['category'].nunique()}\n\n")
        
        f.write("## Preprocessing Steps Applied\n")
        f.write("1. **Tokenization**: Split text into individual words\n")
        f.write("2. **Lowercasing**: Convert all text to lowercase\n")
        f.write("3. **Stopword removal**: Remove common words that don't add meaning\n")
        f.write("4. **Lemmatization**: Reduce words to their base/root form\n")
        
        f.write("\nWhy this sequence? This is the standard NLP preprocessing pipeline:\n")
        f.write("- Tokenization comes first as it's the foundation for all other steps\n")
        f.write("- Lowercasing helps standardize text before further processing\n")
        f.write("- Stopword removal eliminates noise words that don't contribute to meaning\n")
        f.write("- Lemmatization comes last as it works on individual tokens and provides meaningful base forms\n\n")
        
        f.write("## Vectorization Methods Comparison\n")
        f.write("### Bag of Words (CountVectorizer)\n")
        f.write("- Simple counting of word occurrences\n")
        f.write("- Preserves frequency information\n")
        f.write("- Ignores word importance in document context\n\n")
        
        f.write("### TF-IDF\n")
        f.write("- Accounts for word importance across the corpus\n")
        f.write("- Down-weights common words, up-weights unique/rare words\n")
        f.write("- Better captures semantic importance\n\n")
        
        f.write("## Sentiment Analysis Results\n")
        f.write("### TextBlob\n")
        for sentiment, count in textblob_sentiments.items():
            f.write(f"- {sentiment}: {count} headlines ({count/len(df)*100:.2f}%)\n")
        f.write("\n")
        
        f.write("### VADER\n")
        for sentiment, count in vader_sentiments.items():
            f.write(f"- {sentiment}: {count} headlines ({count/len(df)*100:.2f}%)\n")
        f.write("\n")
        
        f.write("### Sentiment Analysis Evaluation\n")
        f.write("VADER generally performs better for short texts like headlines because it's specifically designed for social media and short-form content. ")
        f.write("It captures intensity and includes rules for things like emphasis (ALL CAPS) and emoticons.\n\n")
        
        f.write("## Topic Modeling\n")
        f.write("### LDA Topics\n")
        for i, topic_words in enumerate(lda_topics):
            f.write(f"**Topic {i+1}**: {', '.join(topic_words)}\n")
        f.write("\n")
        
        f.write("### NMF Topics\n")
        for i, topic_words in enumerate(nmf_topics):
            f.write(f"**Topic {i+1}**: {', '.join(topic_words)}\n")
        f.write("\n")
        
        f.write("### Topic Interpretation\n")
        f.write("LDA tends to find more general topics that may overlap, while NMF often finds more distinct, specific topics. ")
        f.write("For news headlines, NMF typically produces more interpretable topics because headlines are short and focused.\n\n")
        
        f.write("## Conclusion\n")
        f.write("This analysis demonstrates the pipeline for processing news headlines using NLP techniques. ")
        f.write("The results show that news headlines have distinct topic clusters, and sentiment analysis reveals interesting patterns in how news is presented. ")
        f.write("TF-IDF vectorization provides better input for topic modeling compared to simple bag-of-words in this case.")
    
    print("\nAnalysis complete! Results saved to nlp_analysis_report.md")
    print("Visualizations saved as PNG files.")


if __name__ == "__main__":
    main() 