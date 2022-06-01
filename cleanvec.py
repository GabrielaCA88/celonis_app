import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import re
import json

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import wordnet
import joblib
import pickle

from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import tokenizer_from_json
#Cleaning and vectorization class


class Cleanvec():
    def __init__(self):
        pass

    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)


    def get_clean_text(self, text):
        pat1 = r'http[^ ]+'  # pattern to remove links
        pat2 = r'www.[^ ]+'  # pattern to remove links
        pat3 = '[^a-zA-Z]+'  # pattern to remove numbers
        pat = r'|'.join((pat1, pat2, pat3))

        filtrar = ['borderlands', 'callofdutyblackopscoldwar', 'amazon', 'overwatch',
                   'xbox(xseries)', 'nba2k', 'dota2', 'playstation5(ps5)',
                   'worldofcraft', 'csgo', 'google', 'assassinscreed', 'apexlegends',
                   'leagueoflegends', 'fortnite', 'microsoft', 'hearthstone',
                   'battlefield', 'playerunknownsbattlegrounds(pubg)', 'verizon',
                   'homedepot', 'fifa', 'reddeadredemption(rdr)', 'callofduty',
                   'tomclancysrainbowsix', 'facebook', 'grandtheftauto(gta)',
                   'maddennfl', 'johnson&johnson', 'cyberpunk2077',
                   'tomclancysghostrecon', 'nvidia']

        text = str(text)
        text = text.lower() #lowercase
        text = re.sub(pat, ' ', text) # remove pat
        text = [w for w in tok.tokenize(text) if not w in stop_words] # remove stop_words
        text = [w for w in text if w not in filtrar]
        text = [w for w in text if len(w)>2] # remove words with len <2
        text = [wordnet_lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in text] # Aplicamos el Lemmatizer
        text = (' '.join(text)).strip() # list to string
        text = text.strip()
        return text

    def vectorized_text (self, text):
        self.get_clean_text(text)
        tfidf_vect = joblib.load("tfidf1.pkl")
        text_vectorized = tfidf_vect.transform(text)
        return text_vectorized

    def padded_text (self, text):
        self.get_clean_text(text)
        print (text)
        with open('text_tokenizer.json') as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
        text_vectorized = tokenizer.texts_to_sequences(text)
        text_padded = pad_sequences(text_vectorized, maxlen=129, padding='post')
        return text_padded
