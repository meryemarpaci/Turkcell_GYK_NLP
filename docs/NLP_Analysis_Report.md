# News Headlines NLP Analysis Report

## Dataset Overview
- Total headlines: 1000
- Categories: 24

## Preprocessing Steps Applied
1. **Tokenization**: Split text into individual words
2. **Lowercasing**: Convert all text to lowercase
3. **Stopword removal**: Remove common words that don't add meaning
4. **Lemmatization**: Reduce words to their base/root form

Why this sequence? This is the standard NLP preprocessing pipeline:
- Tokenization comes first as it's the foundation for all other steps
- Lowercasing helps standardize text before further processing
- Stopword removal eliminates noise words that don't contribute to meaning
- Lemmatization comes last as it works on individual tokens and provides meaningful base forms

## Vectorization Methods Comparison
### Bag of Words (CountVectorizer)
- Simple counting of word occurrences
- Preserves frequency information
- Ignores word importance in document context

### TF-IDF
- Accounts for word importance across the corpus
- Down-weights common words, up-weights unique/rare words
- Better captures semantic importance

## Sentiment Analysis Results
### TextBlob
- Neutral: 661 headlines (66.10%)
- Positive: 202 headlines (20.20%)
- Negative: 137 headlines (13.70%)

### VADER
- Negative: 402 headlines (40.20%)
- Neutral: 341 headlines (34.10%)
- Positive: 257 headlines (25.70%)

### Sentiment Analysis Evaluation
VADER generally performs better for short texts like headlines because it's specifically designed for social media and short-form content. It captures intensity and includes rules for things like emphasis (ALL CAPS) and emoticons.

## Topic Modeling
### LDA Topics
**Topic 1**: say, russian, war, abortion, new, ukrainian, biden, ukraine, covid, oscar
**Topic 2**: week, tweet, funniest, parent, june, court, may, supreme, killed, say
**Topic 3**: say, woman, covid, make, still, show, house, brown, school, man
**Topic 4**: trump, ukraine, russian, say, star, new, york, dead, gop, jan
**Topic 5**: abortion, trump, say, way, new, first, biden, fire, mark, dead

### NMF Topics
**Topic 1**: funniest, tweet, week, june, parent, may, woman, cat, dog, july
**Topic 2**: trump, say, jan, gop, rep, donald, probe, twitter, maralago, panel
**Topic 3**: ukraine, russian, russia, mariupol, war, zelenskyy, say, troop, new, plant
**Topic 4**: court, supreme, abortion, right, roe, biden, wade, law, rule, ruling
**Topic 5**: shooting, school, police, killed, uvalde, texas, say, gov, man, south

### Topic Interpretation
LDA tends to find more general topics that may overlap, while NMF often finds more distinct, specific topics. For news headlines, NMF typically produces more interpretable topics because headlines are short and focused.

## Conclusion
This analysis demonstrates the pipeline for processing news headlines using NLP techniques. The results show that news headlines have distinct topic clusters, and sentiment analysis reveals interesting patterns in how news is presented. TF-IDF vectorization provides better input for topic modeling compared to simple bag-of-words in this case.