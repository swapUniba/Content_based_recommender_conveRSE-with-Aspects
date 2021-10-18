import pandas as pd
from gensim.models import Word2Vec
from gensim import models
from scipy.spatial import distance
import numpy as np


def calculate_centroid(text, model):
    vectors = list()
    for word in text:
        try:
            vector = model.wv.get_vector(word)
            vectors.append(vector)
        except Exception:
            try:
                vector = model[word]
                vectors.append(vector)
            except Exception:
                # print(word, " non c'è")
                continue
    if vectors:
        return np.asarray(vectors).mean(axis=0)
    return np.array([])


def create_model_word2vec(documents, model_name):
    word2vec = Word2Vec(size=100, min_count=1, workers=8)
    word2vec.build_vocab(documents)
    word2vec.train(documents, epochs=100, total_examples=word2vec.corpus_count)
    word2vec.save(model_name)
    return word2vec


def load_model(documents, model_name, pretrained, queue=None):
    if pretrained:
        word2vec = models.KeyedVectors.load_word2vec_format('Models/Word2Vec/word2vec-google-news-300.bin', binary=True)
    else:
        try:
            word2vec = Word2Vec.load(model_name)
        except Exception:
            word2vec = create_model_word2vec(documents, model_name)
    if queue is not None:
        queue.put(word2vec)
    return word2vec


def get_recommendations_word2vec(token_strings, documents, titles, IDs, pretrained, modelWord, prefIDs):
    recommend_movies = []
    num_recommends = len(IDs)
    if modelWord is None:
        modelWord = load_model(documents, "Models\Word2Vec\word2vec_model", pretrained, None)
    cos_sim_s = []
    queries = list()
    for string in token_strings:
        queries.append(calculate_centroid(string, modelWord))
    query = np.asarray(queries).mean(axis=0)
    for i, doc in enumerate(documents):
        films_found = calculate_centroid(doc, modelWord)
        if len(films_found) == 0:
            cos_sim_s.append(0)
            continue
        cos_sim = 1 - distance.cosine(query, films_found)
        cos_sim_s.append(cos_sim)
    # cos_sim_s, titles, IDs = zip(*sorted(zip(sim, titles, IDs), reverse=True))
    rank = 1
    for i in range(num_recommends):
        if len(recommend_movies) == num_recommends:
            break
        # if prefIDs is not None:
        #     if IDs[i] in prefIDs:
        #         print(IDs[i])
        #         continue
        recommend_movies.append({"Rank": rank, "ID": IDs[i], "Value": cos_sim_s[i]})
        rank += 1
    return recommend_movies
