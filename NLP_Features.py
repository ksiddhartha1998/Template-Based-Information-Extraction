import glob
import errno
import nltk
import spacy
from spacy import displacy
from nltk.corpus import stopwords 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import Tree
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import copy
import os
import string
from itertools import chain
from nltk.stem import PorterStemmer
import en_core_web_sm


def read_single_file(filename):
    sent_tokens = []
    f = open(filename,encoding="ascii",errors="ignore")
    temp=f.read()
    temp2=sent_tokenize(temp)
    sent_tokens.extend(temp2)
    f.close()
    return sent_tokens

 
def word_tokenization(sentences):
    #stop_words = set(stopwords.words('english'))
    #all_stops = stop_words | set(string.punctuation)
    #word_tokens = [w for w in word_tokens if not w in all_stops]
    word_tokens=[]
    for i in sentences:
        word_tokens.extend(word_tokenize(i))
    return word_tokens


def word_lemmatization(words):
    
    lemmatize_word = []
    lemmatizer = WordNetLemmatizer()
    for word in words:
        lemmatize_word.append(lemmatizer.lemmatize(word))
    return lemmatize_word

def word_stemmatization(words):       
    stemmatize_word = []
    ps = PorterStemmer()
    for word in words:
        stemmatize_word.append(ps.stem(word))
    return stemmatize_word

def POS_tagging(words):
    POS_tags = []
    POS_tags = nltk.pos_tag(words)
    return POS_tags


def dependency_parsing(sentence):
    dependency_parsed_tree =[]
    en_nlp =spacy.load('en_core_web_sm')
    doc = en_nlp(sentence)
    sent= list(doc.sents)
    for s in sent:
        rootOfSentence = s.root.text
    for token in doc:
        dependency_parsed_tree.append([token.dep_,token.head.text,token.text])
    return dependency_parsed_tree

def display_dependency_parsing(sentence):
    en_nlp =spacy.load('en_core_web_sm')
    doc = en_nlp(sent_tokens[0])

    def to_nltk_tree(node):
        if node.n_lefts + node.n_rights > 0:
            return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
        else:
            return node.orth_
    [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
    #displacy.render(doc, style='dep', jupyter=True)



def wordnet_features(words):
    synonymns_list = []
    hypernyms_list = []
    hyponyms_list = []
    meronyms_list = []
    holonyms_list = []
    for word in words:
        for i,j in enumerate(wn.synsets(word)):
            synonymns_list.extend(wn.synset(j.name()).lemma_names())
            hypernyms_list.extend(list(chain(*[l.lemma_names() for l in j.hypernyms()])))
            hyponyms_list.extend(list(chain(*[l.lemma_names() for l in j.hyponyms()])))
            meronyms_list.extend(list(chain(*[l.lemma_names() for l in j.part_meronyms()])))
            holonyms_list.extend(list(chain(*[l.lemma_names() for l in j.part_holonyms()])))
    return synonymns_list,hypernyms_list,hyponyms_list,meronyms_list,holonyms_list

def named_entity_recognition(sentence):
    entities=[]
    entity_labels=[]
    nlp = en_core_web_sm.load()
    doc = nlp(sentence)
    for X in doc.ents:
        entities.append(X.text)
        entity_labels.append(X.label_)
    return entities,entity_labels



sent_tokens=read_single_file("AbrahamLincoln.txt")


print("Sentence Tokens :")
print(sent_tokens[0:10])

word_tokens=word_tokenization(sent_tokens)
print("Word Tokens :")
print(word_tokens[0:10])

stemma=word_stemmatization(word_tokens)
print("Stemma :")
print(stemma[0:10],word_tokens[0:10])

lema=word_lemmatization(word_tokens)
print("Lemma :")
print(lema[0:10],word_tokens[0:10])

pos=POS_tagging(word_tokens)
print("POS Tags :")
print(pos[0:10],word_tokens[0:10])
print("D-Parse Tree :")
print(dependency_parsing(sent_tokens[0]))

syn,hyper,hypo,mero,holo=wordnet_features(word_tokens)
print("Synonyms :")
print(syn[0:10])
print("Hypernyms :")
print(hyper[0:10])
print("Hyponyms :")
print(hypo[0:10])
print("Meronyms :")
print(mero[0:10])
print("Holonyms :")
print(holo[0:10])
print("NER :")
print(named_entity_recognition(sent_tokens[0]))
