import sys
import gensim.downloader

if len(sys.argv) < 2:
    print("Usage: python vectorPlay.py <word>")
    sys.exit(1)

# Get the word from the command line
input_word = sys.argv[1]

# Load the pre-trained GloVe model
print("Loading model... (this may take a few seconds the first time)")
model = gensim.downloader.load("glove-wiki-gigaword-50")

# Check if word exists in the model
if input_word not in model:
    print(f"'{input_word}' not found in vocabulary.")
    sys.exit(1)

# Get the top 10 most similar words
similar_words = model.most_similar(input_word, topn=10)

# Print them
print(f"\nTop 10 words similar to '{input_word}':\n")
for word, similarity in similar_words:
    print(f"{word}: {similarity:.4f}")
