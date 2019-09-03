import numpy as np
import pandas as pd
import nltk
import re
import networkx as nx
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
#nltk.download('punkt') -- one time run

stop_words = stopwords.words('english')
word_embeddings = {}
clean_sentences = []
#print(stop_words)
def get_data():

    #file_path = str(input("Enter path: "))
    f = open('./Textdata.txt', 'r')
    data = f.read()
    sentences = (sent_tokenize(data))
    print("Got data")
    return sentences

def clean_data():

    sentences = get_data()
    sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    sentences = [s.lower() for s in sentences]

    for sen in sentences:
        sen = sen.split()
        sentence = remove_stopwords(sen)
        clean_sentences.append(sentence)

    print("cleaned sentences")

def remove_stopwords(sentence):

    new_sen = ' '.join([word for word in sentence if word not in stop_words])
    return new_sen

def vectorization():

    clean_data()
    f = open('E:\\Glove\\glove.6B.100d.txt', encoding = 'utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = 'float32')
        word_embeddings[word] = coefs

    f.close()
    print('Glove job done')

    sentence_vectors = []
    for sentence in clean_sentences:

        if len(sentence) != 0:
             v = sum([word_embeddings.get(w, np.zeros((100,))) for w in sentence.split()])/(len(sentence.split())+0.001)
        else:
             v = np.zeros((100,))
        sentence_vectors.append(v)

    print("Done vectorization")
    return sentence_vectors

def similarity_matrix():

    sentence_vectors = vectorization()
    matrix = np.zeros([len(clean_sentences), len(clean_sentences)])

    for i in range(len(clean_sentences)):
        for j in range(len(clean_sentences)):
            matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

    print("Similarity matrix constructed")
    return matrix

def scoring_ranking(num):

    sim_matrix = similarity_matrix()
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(clean_sentences)), reverse=True)

    for i in range(num):
        print(ranked_sentences[i][1])

def extract_sentences():

    number_of_sentences = int(input("Enter the number of sentences required: "))
    scoring_ranking(number_of_sentences)

extract_sentences()
