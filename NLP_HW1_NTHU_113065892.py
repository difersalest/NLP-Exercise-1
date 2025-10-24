# %% [markdown]
# ## Part I: Data Pre-processing

# %%
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import gensim.downloader
import numpy as np
import pandas as pd

# %%
# Download the Google Analogy dataset
!wget http: // download.tensorflow.org/data/questions-words.txt

# %%
# Preprocess the dataset
file_name = "questions-words"
with open(f"{file_name}.txt", "r") as f:
    data = f.read().splitlines()

# %%
# check data from the first 10 entries
for entry in data[:10]:
    print(entry)

# %%
# TODO1: Write your code here for processing data to pd.DataFrame
# Please note that the first five mentions of ": " indicate `semantic`,
# and the remaining nine belong to the `syntatic` category.
count_subcat = 0
count_entries = 0
questions = []
categories = []
sub_categories = []
current_subcat = ""
current_cat = ""
for entry in data:
    count_entries += 1
    if ": " in entry:
        count_subcat += 1
        if (count_subcat > 1):
          print(
              f"Entries for category {current_cat} and subcategory {current_subcat}: \n{count_entries}")
        count_entries = 0
        current_subcat = entry
        if count_subcat <= 5:
            current_cat = "Semantic"
        elif count_subcat > 5:
            current_cat = "Syntactic"
    else:
        categories.append(current_cat)
        sub_categories.append(current_subcat)
        questions.append(entry)

print(
    f"Entries for category {current_cat} and subcategory {current_subcat}: \n{count_entries}")
print(f"\nCount of subcategories: {count_subcat}")


# %%
# Create the dataframe
df = pd.DataFrame(
    {
        "Question": questions,
        "Category": categories,
        "SubCategory": sub_categories,
    }
)

# %%
df.head()

# %%
df.to_csv(f"{file_name}.csv", index=False)

# %% [markdown]
# ## Part II: Use pre-trained word embeddings
# - After finish Part I, you can run Part II code blocks only.

# %%

# %%
data = pd.read_csv("questions-words.csv")

# %%
MODEL_NAME = "glove-wiki-gigaword-100"
# You can try other models.
# https://radimrehurek.com/gensim/models/word2vec.html#pretrained-models

# Load the pre-trained model (using GloVe vectors here)
model = gensim.downloader.load(MODEL_NAME)
print("The Gensim model loaded successfully!")

# %% [markdown]
# Exploring the data manipulation functions

# %%
data["Question"][0].split(" ")[0]

# %%
str(data["Question"][0])

# %%
model[data["Question"][0].split(" ")[0].lower()]

# %%
word_a = model[data["Question"][0].split(" ")[0].lower()]
word_b = model[data["Question"][0].split(" ")[1].lower()]
word_c = model[data["Question"][0].split(" ")[2].lower()]
word_d = model[data["Question"][0].split(" ")[3].lower()]

pred = word_b + word_c - word_a

model.most_similar(positive=pred, topn=1)[0][0]

# %%
# Do predictions and preserve the gold answers (word_D)
preds = []
golds = []

for analogy in tqdm(data["Question"]):
      # TODO2: Write your code here to use pre-trained word embeddings for getting predictions of the analogy task.
      # You should also preserve the gold answers during iterations for evaluations later.
      """ Hints
      # Unpack the analogy (e.g., "man", "woman", "king", "queen")
      # Perform vector arithmetic: word_b + word_c - word_a should be close to word_d
      # Source: https://github.com/piskvorky/gensim/blob/develop/gensim/models/keyedvectors.py#L776
      # Mikolov et al., 2013: big - biggest and small - smallest
      # Mikolov et al., 2013: X = vector(”biggest”) − vector(”big”) + vector(”small”).
      """
      word_a = analogy.split(" ")[0].lower()
      word_b = analogy.split(" ")[1].lower()
      word_c = analogy.split(" ")[2].lower()
      word_d = analogy.split(" ")[3].lower()
      pred = model[word_b] + model[word_c] - model[word_a]
      preds.append(model.most_similar(positive=pred, topn=1)[0][0])
      golds.append(word_d)


# %%
# Perform evaluations. You do not need to modify this block!!

def calculate_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    return np.mean(gold == pred)


golds_np, preds_np = np.array(golds), np.array(preds)
data = pd.read_csv("questions-words.csv")

# Evaluation: categories
for category in data["Category"].unique():
    mask = data["Category"] == category
    golds_cat, preds_cat = golds_np[mask], preds_np[mask]
    acc_cat = calculate_accuracy(golds_cat, preds_cat)
    print(f"Category: {category}, Accuracy: {acc_cat * 100}%")

# Evaluation: sub-categories
for sub_category in data["SubCategory"].unique():
    mask = data["SubCategory"] == sub_category
    golds_subcat, preds_subcat = golds_np[mask], preds_np[mask]
    acc_subcat = calculate_accuracy(golds_subcat, preds_subcat)
    print(f"Sub-Category{sub_category}, Accuracy: {acc_subcat * 100}%")

# %% [markdown]
# Exploring the data before plotting

# %%
data[data['SubCategory'] == ": family"].iloc[:]["Question"]

# %%
# Collect words from Google Analogy dataset
SUB_CATEGORY = ": family"

# TODO3: Plot t-SNE for the words in the SUB_CATEGORY `: family`

# Definint the t-SNE 2D representation
tsne = TSNE(n_components=2, metric='cosine', random_state=42)

family_vect_data = []
# words_to_plot = []
words_to_plot = set()

# Obtaining all the unique words from the analogies in the 'family' subcategory
for analogy in data[data['SubCategory'] == ": family"].iloc[:]["Question"]:
    for word in analogy.split(" "):
        words_to_plot.add(word.lower())
        # This part uncommented obtains all words, even non-unique
        # words_to_plot.append(word.lower())
        # family_vect_data.append(model[word.lower()])

# Obtaining the embedding vectors of the unique words in the subcategory
for word in words_to_plot:
    family_vect_data.append(model[word.lower()])

# Obtaining the tsne 2D representation of the embeddings
converted_vec = tsne.fit_transform(np.array(family_vect_data))


# Plotting the figure
plt.figure(figsize=(10, 10), dpi=100)
plt.scatter(converted_vec[:, 0], converted_vec[:, 1])

# Labeling each point with the related word
for label, x, y in zip(words_to_plot, converted_vec[:, 0], converted_vec[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0),  textcoords='offset points')
plt.title("Word Relationships from Google Analogy Task")
plt.show()
plt.savefig("word_relationships.png", bbox_inches="tight")

# %% [markdown]
# ### Part III: Train your own word embeddings

# %% [markdown]
# ### Get the latest English Wikipedia articles and do sampling.
# - Usually, we start from Wikipedia dump (https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2). However, the downloading step will take very long. Also, the cleaning step for the Wikipedia corpus ([`gensim.corpora.wikicorpus.WikiCorpus`](https://radimrehurek.com/gensim/corpora/wikicorpus.html#gensim.corpora.wikicorpus.WikiCorpus)) will take much time. Therefore, we provide cleaned files for you.

# %%
# Download the split Wikipedia files
# Each file contain 562365 lines (articles).
!gdown - -id 1jiu9E1NalT2Y8EIuWNa1xf2Tw1f1XuGd - O wiki_texts_part_0.txt.gz
!gdown - -id 1ABblLRd9HXdXvaNv8H9fFq984bhnowoG - O wiki_texts_part_1.txt.gz
!gdown - -id 1z2VFNhpPvCejTP5zyejzKj5YjI_Bn42M - O wiki_texts_part_2.txt.gz
!gdown - -id 1VKjded9BxADRhIoCzXy_W8uzVOTWIf0g - O wiki_texts_part_3.txt.gz
!gdown - -id 16mBeG26m9LzHXdPe8UrijUIc6sHxhknz - O wiki_texts_part_4.txt.gz

# %%
# Download the split Wikipedia files
# Each file contain 562365 lines (articles), except the last file.
!gdown - -id 17JFvxOH-kc-VmvGkhG7p3iSZSpsWdgJI - O wiki_texts_part_5.txt.gz
!gdown - -id 19IvB2vOJRGlrYulnTXlZECR8zT5v550P - O wiki_texts_part_6.txt.gz
!gdown - -id 1sjwO8A2SDOKruv6-8NEq7pEIuQ50ygVV - O wiki_texts_part_7.txt.gz
!gdown - -id 1s7xKWJmyk98Jbq6Fi1scrHy7fr_ellUX - O wiki_texts_part_8.txt.gz
!gdown - -id 17eQXcrvY1cfpKelLbP2BhQKrljnFNykr - O wiki_texts_part_9.txt.gz
!gdown - -id 1J5TAN6bNBiSgTIYiPwzmABvGhAF58h62 - O wiki_texts_part_10.txt.gz

# %% [markdown]
# Couldn't download with the above links, so I tried the others.

# %%
!gdown - -id 1J0os1846PQ129t720aI0wMm-5GepEwSl -O wiki_texts_part_0.txt.gz
!gdown --id 1tsI3RSKPN3b2-1IZ0N7bmjgVRf-THIkW -O wiki_texts_part_1.txt.gz
!gdown --id 1koiw6RFNzDe6pe2zMTfVhsEKmpmnYyu5 -O wiki_texts_part_2.txt.gz
!gdown --id 1YSGbDqhbg2xJsWD_hYQ5z9URl0dCTC2m -O wiki_texts_part_3.txt.gz
!gdown --id 1PA3C99C8CcLFjkenT0a9iU07XEQmXyG_ -O wiki_texts_part_4.txt.gz

# %%
!gdown --id 1sSLea4hq6Z7oT6noOU_II1ahWjNOKcDX -O wiki_texts_part_5.txt.gz
!gdown --id 1i6kXTDtZkRiivJ0mj-5GkVbE4gMFlmSb -O wiki_texts_part_6.txt.gz
!gdown --id 1ain2DN1nxXfsmJ2Aj9TFZlLVJSPsu9Jb -O wiki_texts_part_7.txt.gz
!gdown --id 1UKhvielQDqQz5pMZ7J3SHv9m8_8gO-dE -O wiki_texts_part_8.txt.gz
!gdown --id 1q1zMA4hbMS7tID2GTQx-c94UPB8YQaaa -O wiki_texts_part_9.txt.gz
!gdown --id 1-kkGxwMxPsoGg5_2pdaOeE3Way6njLpH -O wiki_texts_part_10.txt.gz

# %%
# Extract the downloaded wiki_texts_parts files.
!gunzip -k wiki_texts_part_*.gz

# %%
# Combine the extracted wiki_texts_parts files.
!cat wiki_texts_part_*.txt > wiki_texts_combined.txt

# %%
# Check the first ten lines of the combined file
!head -n 10 wiki_texts_combined.txt

# %% [markdown]
# Please note that we used the default parameters of [`gensim.corpora.wikicorpus.WikiCorpus`](https://radimrehurek.com/gensim/corpora/wikicorpus.html#gensim.corpora.wikicorpus.WikiCorpus) for cleaning the Wiki raw file. Thus, words with one character were discarded.

# %%
# Now you need to do sampling because the corpus is too big.
# You can further perform analysis with a greater sampling ratio.

import random
from tqdm import tqdm

wiki_txt_path = "wiki_texts_combined.txt"
# wiki_texts_combined.txt is a text file separated by linebreaks (\n).
# Each row in wiki_texts_combined.txt indicates a Wikipedia article.

# Defining variables
output_path = "wiki_texts_combined_sample_20_prcnt.txt"
article_list = []
sample_percentage = 0.2
with open(wiki_txt_path, "r", encoding="utf-8") as f:
    # We obtain the length of the articles
    total=sum(1 for line in f)
    print(f"Total articles: {total}")
    f.seek(0) # after counting moving pointer to the beginning
    with open(output_path, "w", encoding="utf-8") as output_file:
      # TODO4: Sample `20%` Wikipedia articles
      # Write your code here

      # We obtain the length of the sample
      sample_len = sample_percentage * total
      print(f"\nSample length of articles ({sample_percentage*100})%: {sample_len}")
      sampled_indices_articles = random.sample(range(total), int(sample_len))

      # I was having problems processing the lines of the articles fast
      # So I asked Gemini what is the bottleneck in my code, and it was that before I was using a list
      # to validate the sampled_indices_articles, but Gemini suggested to use a set instead for faster lookup

      # for lookup we use a set
      sampled_indices_set = set(sampled_indices_articles)
      print(f"Length sampled indices {len(sampled_indices_articles)}")
      for index, line in tqdm(zip(range(total), f), desc="Writing sampled articles to file...", total=total):
        if index in sampled_indices_set:
          output_file.write(line)


# %%
# TODO5: Train your own word embeddings with the sampled articles
# https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
# Hint: You should perform some pre-processing before training.
# Preprocessing
sample_article_lst = []
sample_path = "wiki_texts_combined_sample_20_prcnt.txt"
with open(sample_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Appending sampled articles..."):
      sample_article_lst.append(line)

# %%
len(sample_article_lst)

# %%
# Converting the list into pandas dataframe
import pandas as pd
sample_articles_df = pd.DataFrame(
    {
        "Articles": sample_article_lst,
    }
)

# %%
from fast_langdetect import detect

print(detect("Hello, world!", model="auto", k=1))

# %%
detect("Hello, world!", model="auto", k=1)[0]['lang']

# %%
from fast_langdetect import detect

def detect_english(text: str)->str:
    return detect(text, model="auto", k=1)[0]['lang']

print(detect_english("Hello my friend!"))

# %%
tqdm.pandas()
# Gemini helped me in articulating the lambda function
sample_articles_df['lang'] = sample_articles_df['Articles'].apply(lambda x: detect_english(x))

# Filter for English articles
english_articles_df = sample_articles_df[sample_articles_df['lang'] == 'en'].copy()
print(f"Found {len(english_articles_df)} English articles.")

# %%
english_articles_df["Tokenized_Articles"] = None
english_articles_df

# %%
english_articles_df.to_pickle("sampled_20_prcnt_english_articles.pkl")

# %%
import pandas as pd
english_articles_df = pd.read_pickle('sampled_20_prcnt_english_articles.pkl')

# %%
from tqdm import tqdm
import spacy

spacy.require_gpu() # To use GPU
nlp = spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'ner', 'senter', 'attribute_ruler'], exclude=['tagger', 'parser', 'ner', 'senter', 'attribute_ruler'])
nlp.add_pipe("doc_cleaner", last=True) 

def validate_pipe(doc):
    lemma_list = [token.lemma_.lower() for token in doc if not token.is_stop]
    return lemma_list

def preprocess_pipe(texts, batch_size=110):
    processed_docs = []
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size), desc="Tokenizing articles...", total = int(len(texts)/batch_size)):
        processed_docs.append(validate_pipe(doc))

    return processed_docs

# %%
english_articles_df["Tokenized_Articles"] = preprocess_pipe(english_articles_df["Articles"])
# preprocess_pipe(english_articles_df["Articles"])

# %%
english_articles_df.to_pickle("sampled_20_prcnt_tknzd_eng_articles.pkl")

# %%
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(filename='gensim.log', format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO, force=True)

class MonitorCallback(CallbackAny2Vec):
    def __init__(self, test_words):
        self._test_words = test_words

    def on_epoch_end(self, model):
        print("Model loss:", model.get_latest_training_loss())  # print loss
        # for word in self._test_words:  # show wv logic changes
        #     print(model.wv.most_similar(word))
def get_unique_words(tokenized_articles):
    unique_words = set()
    for article in tqdm(tokenized_articles, desc="Counting unique words..."):
        for word in article:
            unique_words.add(word)
    return unique_words

unique_words = list(get_unique_words(english_articles_df["Tokenized_Articles"]))
print(f"Amount of unique words: {len(unique_words)}")
monitor = MonitorCallback(["happy", "angry", "news"])
# setting
vector_dim = 100
window_size = 5
min_count = 20
training_epochs = 1
cpu_cores = 32

# model
word2vec_model = Word2Vec(sentences=english_articles_df["Tokenized_Articles"], 
                          vector_size=vector_dim, window=window_size, workers=cpu_cores,
                          min_count=min_count, epochs=training_epochs, callbacks=[monitor], compute_loss=True)

# %%
word2vec_model.save("wiki_sample_20_prcnt_model")

# %%
# If we want to load
word2vec_model = Word2Vec.load("wiki_sample_20_prcnt_model")

# %%
data = pd.read_csv("questions-words.csv")

# %%
import pandas as pd
# Do predictions and preserve the gold answers (word_D)
preds = []
golds = []

data_predictions = data.copy()
data_predictions["Gold"] = None
data_predictions["Prediction"] = None

for index, analogy in tqdm(enumerate(data["Question"]), total = len(data["Question"])):
      # TODO6: Write your code here to use your trained word embeddings for getting predictions of the analogy task.
      # You should also preserve the gold answers during iterations for evaluations later.
      """ Hints
      # Unpack the analogy (e.g., "man", "woman", "king", "queen")
      # Perform vector arithmetic: word_b + word_c - word_a should be close to word_d
      # Source: https://github.com/piskvorky/gensim/blob/develop/gensim/models/keyedvectors.py#L776
      # Mikolov et al., 2013: big - biggest and small - smallest
      # Mikolov et al., 2013: X = vector(”biggest”) − vector(”big”) + vector(”small”).
      """
      word_a = analogy.split(" ")[0].lower()
      word_b = analogy.split(" ")[1].lower()
      word_c = analogy.split(" ")[2].lower()
      word_d = analogy.split(" ")[3].lower()
      try:
            pred = word2vec_model.wv[word_b] + word2vec_model.wv[word_c] - word2vec_model.wv[word_a]
            most_similar = word2vec_model.wv.most_similar(positive = pred, topn = 1)[0][0]
            preds.append(most_similar)
            golds.append(word_d)
            data_predictions.loc[data_predictions.index[index], "Prediction"] = most_similar
            data_predictions.loc[data_predictions.index[index], "Gold"] = word_d
      except Exception as e:
            print(f"Couldn't predict the question: {e}")
            data_predictions.loc[data_predictions.index[index], "Prediction"] = None
            data_predictions.loc[data_predictions.index[index], "Gold"] = None

# %%
# There were many questions that could not be answer this time due to either the word being filtered as it is considered a stopword
# Or maybe because it was infrequent (filtered min freq of 20) or because it was not inside the sampled part of the corpus

print(f"Total amount of data predicted: {len(preds)}/{len(data["Question"])}")

# %%
# Gemini helped me code this to get the data on the Nones per category and subcategory: 
for category in data_predictions["Category"].unique():
    masked_df = data_predictions.loc[data_predictions["Category"]==category]
    
    # Count of Nones for the category
    nones_per_category = masked_df['Prediction'].isnull().sum()
    print(f"Category: {category}, Count of Nones: {nones_per_category}")

for subcategory in data_predictions["SubCategory"].unique():
    masked_df = data_predictions.loc[data_predictions["SubCategory"]==subcategory]
    
    # Count of Nones for the category
    nones_per_subcategory = masked_df['Prediction'].isnull().sum()
    print(f"Sub-Category{subcategory}, Count of Nones: {nones_per_subcategory}")

# %%
data_predictions

# %%
none_counts_per_column = data_predictions.isna().sum()
print("None values per column:")
print(none_counts_per_column)

# %%
import numpy as np
# Perform evaluations. You do not need to modify this block!!

def calculate_accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    return np.mean(gold == pred)

golds_np, preds_np = np.array(golds), np.array(preds)
# data = pd.read_csv("questions-words.csv")

# Evaluation: categories
for category in data_predictions["Category"].unique():
    # mask = data_predictions["Category"] == category 
    masked_df = data_predictions.loc[data_predictions["Category"]==category]
    # golds_cat = masked_df.loc[~masked_df["Gold"].isna()].to_numpy()
    # preds_cat = masked_df.loc[~masked_df["Prediction"].isna()].to_numpy()
    golds_cat = masked_df["Gold"].to_numpy()
    preds_cat = masked_df["Prediction"].to_numpy()
    # golds_cat, preds_cat = golds_np[mask], preds_np[mask]
    acc_cat = calculate_accuracy(golds_cat, preds_cat)
    print(f"Category: {category}, Accuracy: {acc_cat * 100}%")

# Evaluation: sub-categories
for sub_category in data_predictions["SubCategory"].unique():
    # mask = data_predictions["SubCategory"] == sub_category
    # golds_subcat, preds_subcat = golds_np[mask], preds_np[mask]
    masked_df = data_predictions.loc[data_predictions["SubCategory"]==sub_category]
    golds_subcat = masked_df["Gold"].to_numpy()
    preds_subcat = masked_df["Prediction"].to_numpy()
    acc_subcat = calculate_accuracy(golds_subcat, preds_subcat)
    print(f"Sub-Category{sub_category}, Accuracy: {acc_subcat * 100}%")

# %%
# Collect words from Google Analogy dataset
SUB_CATEGORY = ": family"

# TODO7: Plot t-SNE for the words in the SUB_CATEGORY `: family`
from sklearn.manifold import TSNE

# Definint the t-SNE 2D representation
tsne = TSNE(n_components=2, metric='cosine', random_state=42)

family_vect_data = []
words_to_plot = set()

# Obtaining all the unique words from the analogies in the 'family' subcategory
for analogy in data[data['SubCategory'] == ": family"].iloc[:]["Question"]:
    for word in analogy.split(" "):
        words_to_plot.add(word.lower())

# Obtaining the embedding vectors of the unique words in the subcategory
for word in words_to_plot:
    # Some words are not in the vocabulary, so I needed to filter them
    try:
        family_vect_data.append(word2vec_model.wv[word.lower()])
    except Exception as e:
        print(f"Couldn't get the embeddings: {e}")

# Obtaining the tsne 2D representation of the embeddings
converted_vec = tsne.fit_transform(np.array(family_vect_data))


# Plotting the figure
plt.figure(figsize=(10, 10), dpi=100)
plt.scatter(converted_vec[:, 0], converted_vec[:, 1])

# Labeling each point with the related word
for label, x, y in zip(words_to_plot, converted_vec[:, 0], converted_vec[:, 1]):
    plt.annotate(label, xy=(x,y), xytext=(0,0),  textcoords='offset points')
plt.title("Word Relationships from Google Analogy Task")
plt.show()
plt.savefig("word_relationships_new_model.png", bbox_inches="tight")


