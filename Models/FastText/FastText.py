import fasttext.util
import numpy as np
import pandas as pd
from gensim.models import FastText
from scipy.spatial import distance


def calculate_centroid(text, model):
    vectors = list()
    if len(text) == 0:
        return [-1]
    for word in text:
        try:
            vector = model.wv.get_vector(word)
            vectors.append(vector)
        except Exception:
            # print(word, " non c'è")
            continue
    if vectors:
        return np.asarray(vectors).mean(axis=0)
    return np.array([])


def centroid_fastext_FB(text, model):
    vector_string = list()
    if len(text) == 0:
        return [-1]
    for token in text:
        res = model.get_word_vector(token)
        vector_string.append(res)
        if len(vector_string) == 0:
            # print(token, " non c'è")
            vector_string.append(0)
    return np.asarray(vector_string).mean(axis=0)


def create_model_fasttext_fb(model, queue=None):
    # fasttext.util.download_model('en', if_exists='ignore')  # English
    model = fasttext.load_model('Models/FastText/cc.en.300.bin')
    if queue is not None:
        queue.put(model)
    return model


def create_model_fasttext(documents, model_name):
    fasttext = FastText(min_count=1, workers=8, size=100)
    fasttext.build_vocab(documents)
    fasttext.train(documents, total_examples=fasttext.corpus_count, epochs=100)
    fasttext.save(model_name)
    return fasttext


def load_model(documents, model_name, queue=None):
    try:
        fasttext = FastText.load(model_name)
    except Exception:
        fasttext = create_model_fasttext(documents, model_name)
    if queue is not None:
        queue.put(fasttext)
    return fasttext


def get_recommendations_fastText(token_strings, documents, titles, IDs, modelFastText, pretrained, prefIDs):
    cos_sim_s = []
    recommend_movies = []
    num_recommends = len(IDs)
    if modelFastText is None:
        if pretrained:
            modelFastText = create_model_fasttext_fb(modelFastText, None)
        else:
            modelFastText = load_model(documents, "Models/FastText/fasttext_model", None)
    if pretrained:
        queries = list()
        for string in token_strings:
            queries.append(centroid_fastext_FB(string, modelFastText))
        query = np.asarray(queries).mean(axis=0)
    else:
        queries = list()
        for string in token_strings:
            queries.append(calculate_centroid(string, modelFastText))
        query = np.asarray(queries).mean(axis=0)
    for i, doc in enumerate(documents):
        if pretrained:
            films_found = centroid_fastext_FB(doc, modelFastText)
        else:
            films_found = calculate_centroid(doc, modelFastText)
        if films_found[0] == -1:
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
