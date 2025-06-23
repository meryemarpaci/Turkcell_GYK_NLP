# News Headlines NLP Analysis

## Project Overview
This project implements various Natural Language Processing (NLP) techniques on news headlines data to perform:
1. Text preprocessing
2. Vectorization methods comparison
3. Sentiment analysis
4. Topic modeling

## Dataset
The project uses the "News Category Dataset" from Kaggle, which contains news headlines with their categories, descriptions, and other metadata. The dataset structure is as follows:

```json
{
  "link": "https://www.huffpost.com/entry/covid-boosters-uptake-us_n_632d719ee4b087fae6feaac9",
  "headline": "Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID Boosters",
  "category": "U.S. NEWS",
  "short_description": "Health experts said it is too early to predict whether demand would match up with the 171 million doses of the new boosters the U.S. ordered for the fall.",
  "authors": "Carla K. Johnson, AP",
  "date": "2022-09-23"
}
```

## Requirements
To run this project, you need the following Python packages:
```
pandas
numpy
matplotlib
seaborn
nltk
scikit-learn
textblob
wordcloud (optional, for visualization)
```

You can install them using:
```
pip install -r requirements.txt
```

## Project Structure
- `news_nlp_analysis.py` - The main Python script containing all NLP analysis code
- `requirements.txt` - Required Python packages
- `nlp_analysis_report.md` - Generated report with analysis results

## How to Run
1. Ensure the dataset is in the `archive/News_Category_Dataset_v3.json` path
2. Run the main script:
```
python news_nlp_analysis.py
```

## Features Implemented

### 1. Text Preprocessing
- Tokenization
- Lowercasing
- Stopword removal
- Lemmatization
- Optional POS tagging

### 2. Vectorization Methods
- Bag of Words (CountVectorizer)
- TF-IDF Vectorization

### 3. Sentiment Analysis
- TextBlob based sentiment analysis
- VADER sentiment analysis
- Comparison between both methods

### 4. Topic Modeling
- Latent Dirichlet Allocation (LDA)
- Non-negative Matrix Factorization (NMF)
- Topic visualization with WordCloud

## Output Files
- `category_distribution.png` - Visualization of news categories distribution
- `sentiment_distribution.png` - Visualization of sentiment analysis results
- `lda_topic_*.png` - Visualizations of LDA topics
- `nmf_topic_*.png` - Visualizations of NMF topics
- `nlp_analysis_report.md` - A comprehensive report of all analysis results

## Analysis Process
1. Load and explore the dataset
2. Apply preprocessing steps with detailed explanations
3. Compare vectorization methods with examples
4. Perform and compare sentiment analysis methods
5. Implement topic modeling and visualize results
6. Generate a detailed analysis report 