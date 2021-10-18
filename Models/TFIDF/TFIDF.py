import numpy as np
from gensim.similarities import MatrixSimilarity
from gensim.models import TfidfModel
from gensim.corpora import Dictionary


def create_model_tfidf_model(documents, model_name, matrix_name, dic_name):
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    tfidfmodel = TfidfModel(corpus)
    index = MatrixSimilarity(tfidfmodel[corpus], num_features=len(dictionary))
    index.save(matrix_name)
    tfidfmodel.save(model_name)
    dictionary.save(dic_name)
    return tfidfmodel, index, dictionary


def load_model(documents, model_name, matrix_name, dic_name, queue=None):
    try:
        tfidfmodel = TfidfModel.load(model_name)
        index = MatrixSimilarity.load(matrix_name)
        dictionary = Dictionary.load(dic_name)
    except Exception:
        tfidfmodel, index, dictionary = create_model_tfidf_model(documents=documents, model_name=model_name,
                                                                 matrix_name=matrix_name, dic_name=dic_name)
    if queue is not None:
        queue.put([tfidfmodel, index, dictionary])
    return tfidfmodel, index, dictionary


def get_recommendations_tfidf(token_strings, documents, titles, IDs, dictionary, tfidfmodel, index, prefIDs):
    recommend_movies = []
    num_recommends = len(IDs)
    if dictionary is None or tfidfmodel is None or index is None:
        tfidfmodel, index, dictionary = load_model(documents, "Models/TFIDF/tfidf_model",
                                                   "Models/TFIDF/matrix_tfidf",
                                                   "Models/TFIDF/dictionary_tfidf", None)
    sims = []
    try:
        for string in token_strings:
            query = dictionary.doc2bow(string)
            vec_bow_tfidf = tfidfmodel[query]
            sim = index.get_similarities(vec_bow_tfidf)
            sims.append(sim)
    except Exception:
        query = dictionary.doc2bow(token_strings)
        vec_bow_tfidf = tfidfmodel[query]
        sim = index.get_similarities(vec_bow_tfidf)
        sims.append(sim)
    sim = np.asarray(sims).mean(axis=0)
    # cos_sim_s, titles, IDs = zip(*sorted(zip(sim, titles, IDs), reverse=True))
    rank = 1
    for i in range(num_recommends):
        if len(recommend_movies) == num_recommends:
            break
        # if prefIDs is not None:
        #     if IDs[i] in prefIDs:
        #         print(IDs[i])
        #         continue
        recommend_movies.append({"Rank": rank, "ID": IDs[i], "Value": sim[i]})
        rank += 1
    return recommend_movies