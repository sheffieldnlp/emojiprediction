# emojiprediction
Emoji Prediction

* Jan 30 and Jan 31

Approaches tried:
- BOW (pretty poor results)
- ngram based classification (okayish)
- fastText + StarSpace based classification
- get embedding using fastText -> -minCount=2
- use these embeddings to get oovs (fastText)
- LSTM based classification: initialize with embeddings
- Sentence embedding with LSTM for classification
- Sentence embedding is averaged sum of tokens in the sentence
- LSTM based Visual-Semantic Embeddings with Hard Negatives (did not work as
  well)
