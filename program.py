import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import string 
from collections import Counter
import re
from heapq import nlargest
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from nltk.corpus import wordnet
nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')

class Clasa:
    
    def __init__(self):
        self.stemmer = nltk.stem.PorterStemmer()
        pass

    def count_sentences(self, text):
        sentences = nltk.sent_tokenize(text)
        num_sentences = len(sentences)
        return num_sentences

    def tokenize_words(self, text):
        words = nltk.word_tokenize(text)
        wordss = len(words)
        return wordss

    def top_words(self, text, n=5):
       punctuations = string.punctuation

        # Îndepărtează semnele de punctuație din text
       text_without_punctuation = text.translate(str.maketrans('', '', punctuations))

        # Tokenizăm textul fără semne de punctuație
       words = nltk.word_tokenize(text_without_punctuation)

        # Calculăm frecvența cuvintelor
       frequency_distribution = nltk.FreqDist(words)

        # Extragem cele mai comune n cuvinte
       top_words = frequency_distribution.most_common(n)

       return [word for word, _ in top_words]

    def lung(self, text):
        
        return len(text)

    def faraSpatii(self, text):
        
        return len(text.replace(" ", ""))

    def find_plural_words(self, text):
        words = nltk.word_tokenize(text)
        plural_words = [word for word in words if word.endswith(('s', 'es', 'ies'))]
        return len(plural_words)

    def rezumat(self, text, num_sentences=3):
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return ' '.join([str(sentence) for sentence in summary])

    def find_words_with_same_stem(self, text, target_word):
        words = nltk.word_tokenize(text)
        stem = self.stemmer.stem(target_word)
        words_with_same_stem = [word for word in words if self.stemmer.stem(word) == stem]
        return words_with_same_stem

    def prepozitii(self,text):
        tagged_words = nltk.pos_tag(nltk.word_tokenize(text))
        prepositions = [word for word, pos in tagged_words if pos == 'IN']
        return prepositions
    
    def conjunctiile(self,text):
        words = nltk.pos_tag(nltk.word_tokenize(text))
        conjunctions = [word for word, pos in words if pos == 'CC']
        return conjunctions

    def find_antonyms(self, word):
        antonyms = {}
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonyms.append(lemma.antonyms()[0].name().split(".")[0])
        return antonyms

    def find_antonyms_for_top_words(self, text, n=5):
        top_words = [word for word, _ in self.top_words(text, n)]  
        antonyms_dict = {}
        for word in top_words:
            antonyms = self.find_antonyms(word)
            antonyms_dict[word] = antonyms
        return antonyms_dict

    def process_file(self):
        with open('text.txt', 'r') as file:
            text = file.read()
        target_word = "ne"
        result = {}
        
        result["Propozitii"] = self.count_sentences(text)
        result["Lungimea textului"] = self.lung(text)
        result["Lungimea textului fara spatii"] = self.faraSpatii(text)
        result["Cuvinte"] = self.tokenize_words(text)
        result["Cele 5 cuvinte"] = self.top_words(text)
        result["Cuvinte la plural"] = self.find_plural_words(text)
        result["Rezumat "] = self.rezumat(text)
        result["Cuvinte cu aceiasi radacina "] = self.find_words_with_same_stem(text,target_word)
        result["prepozitiile "] = self.prepozitii(text)
        result["Conjunctiile "] = self.conjunctiile(text)
        result["Antonimele "] = self.find_antonyms_for_top_words(text)


        return result

clasa = Clasa()

# Procesează fișierul și obține rezultatele
result = clasa.process_file()

# Scrie rezultatele într-un fișier de ieșire
with open('output.txt', 'w') as file:
    for key, value in result.items():
        file.write(f"{key}: {value}\n")

print("Rezultatele au fost scrise în output.txt")
