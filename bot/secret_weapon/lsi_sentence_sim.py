from __future__ import division
from gensim import corpora, models, similarities
import time
import jieba


datapath = 'bot/datasets/bot_cut_question.txt'
jieba.load_userdict("bot/user_disease_hyper_dict/my_hyper_dict.txt")


class Myheap:
    lst = []
    def __init__(self, lst):
        self.lst = list(lst)

    def get_top_ten(self):
        if len(self.lst) <= 10:
            self.lst.sort(key=lambda doc: doc[1], reverse=True)
            return

        else:
            for start in range(9, -1, -1):
                self.sift_down(start, 9)
            for index in range(10, len(self.lst) - 1, 1):
                if self.lst[index][1] > self.lst[0][1]:
                    self.lst[0] = self.lst[index]
                    self.sift_down(0, 9)
        return sorted(self.lst[:10], key=lambda x: x[1], reverse=True)

    def sift_down(self, start, end):
        root = start
        while True:
            child = 2 * root + 1
            if child > end:
                break
            if child + 1 <= end and self.lst[child][1] > self.lst[child + 1][1]:
                child += 1
            if self.lst[root][1] > self.lst[child][1]:
                self.lst[root], self.lst[child] = self.lst[child], self.lst[root]
                root = child
            else:
                break


def load_data(path):
    corp = []
    with open(path, 'r') as fh:
        for l in fh:
            corp.append(l.split())
    return corp

def lsi_sim(corp):
    dictionary = corpora.Dictionary(corp)
    dictionary.save('bot/brain/simmodel/dictionary.dict')
    corpus = [dictionary.doc2bow(text) for text in corp]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary)
    lsi.save('bot/brain/simmodel/model.lsi')
    index = similarities.MatrixSimilarity(lsi[corpus])
    index.save('bot/brain/simmodel/sim.index')

def lsi_model():
    corp = load_data(datapath)
    lsi_sim(corp)

def lsi_sim_sentence(sentencce):
    corp = load_data(datapath)
    #lsi_sim(corp)
    dictionary = corpora.Dictionary.load('bot/brain/simmodel/dictionary.dict')
    lsi = models.LsiModel.load('bot/brain/simmodel/model.lsi')
    index = similarities.MatrixSimilarity.load('bot/brain/simmodel/sim.index')
    print ('模型加载完毕')
    while True:
        query = ' '.join(jieba.lcut(sentencce)).split()
        time1 = time.time()
        query_bow = dictionary.doc2bow(query)
        query_lsi = lsi[query_bow]
        sims = index[query_lsi]
        # sims = sorted(enumerate(sims), key=lambda item: -item[1])
        simList = zip(range(len(sims)), sims)
        # heap sort
        heap = Myheap(simList)
        result = heap.get_top_ten()
        time2 = time.time()
        print ('Cost %f seconds' % (time2 - time1))
        print ('与它话题相似度前十的句子排行及其相似度如下：')
        for line, sim in result:
            print (' '.join(corp[line]))
            print (sim)
            #return top 1
            return (''.join(corp[line]))

if __name__ == '__main__':
    lsi_model()
    sentencce = '失眠会引起高血压吗？'
    print(lsi_sim_sentence(sentencce))
