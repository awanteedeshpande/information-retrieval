# "main" file to run the code

from baseline import TF_IDF, Metric
from bm25 import Bm25Model
from bm25_sentence import BM25SentenceModel
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    # TF-IDF calculation
    metric = Metric()
    doc_obj = TF_IDF()
    bm25_model = Bm25Model()
    bm25_sent = BM25SentenceModel()

    ranking = {}
    query = metric.get_query_list()
    # rel_doc = get_relevant_doc(pattern, doc_dict) 
    mean_precision_tf = []
    mean_precision_bm25 = []
    mean_precision_sentence = []
    rank = []

    for q_details in query:
        pred_tf = doc_obj.top_baseline(q_details.get("query",""), 1000)
        # ranking[q_details.get("index")] = pred
        precision_tf = metric.check_relevance(q_details, pred_tf, doc_obj)
        mean_precision_tf.append(precision_tf)

        # BM25
        top_1000 = bm25_model.top_1000_doc_dict(pred_tf)
        doc_sim = bm25_model.fit_model(top_1000, q_details.get("query",""))
        pred_bm25 = sorted(doc_sim, key=doc_sim.get, reverse=True)[:50]
        precision_bm25 = metric.check_relevance(q_details, pred_bm25, doc_obj)
        mean_precision_bm25.append(precision_bm25)

        # Sentence ranking
        sent_word_db = bm25_sent.convert_to_sentence(pred_bm25)
        sent_sim = bm25_sent.fit_model(sent_word_db, q_details.get("query",""))
        pred_bm25_sent = sorted(sent_sim, key=sent_sim.get, reverse=True)[:50]
        ranking, precision_sentence = metric.rank(q_details, pred_bm25_sent, sent_word_db)
        mean_precision_sentence.append(precision_sentence)
        print(q_details.get("query"), precision_tf, precision_bm25, precision_sentence, ranking)
        rank.append(ranking)

    # Evaluation based on mean reciprocal rank
    mrr = metric.MRR(rank)
    print("Mean precision of TF-IDF is ", str(sum(mean_precision_tf)/len(mean_precision_tf)))
    print("Mean precision of BM25 is ", str(sum(mean_precision_bm25)/len(mean_precision_bm25)))
    print("Mean precision of Sentence Ranker is ", str(sum(mean_precision_sentence)/len(mean_precision_sentence)))
    print("MRR of BM25 on sentence is", str(mrr))

    # Plot graphs for illustration
    barWidth = 0.35

    r1 = np.arange(len(mean_precision_tf))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    plt.bar(r1, mean_precision_tf, width=barWidth, label='tf-idf')
    plt.bar(r2, mean_precision_bm25, color='green', width=barWidth, label='bm25')
    plt.bar(r3, mean_precision_sentence, color='yellow', width=barWidth, label='sentence')

    plt.xlabel('Query number', fontweight='bold')
    plt.ylabel('Precision')
    plt.xticks([r + barWidth for r in range(len(mean_precision_tf))], range(1, len(query)+1), rotation=45, fontsize=4)
    plt.legend()
    plt.show()
    plt.savefig('Precision_Comparison_All')

    plt.bar(list(range(1, len(query)+1)), rank)
    plt.xlabel('Query number', fontweight='bold')
    plt.ylabel('Rank@1')
    plt.xticks([r + barWidth for r in range(0, len(rank))], range(1, len(query)+1), rotation=45, fontsize=4)
    plt.show()
    plt.savefig('RankVsQuery')