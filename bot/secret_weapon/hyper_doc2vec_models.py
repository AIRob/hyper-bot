# -*- coding: utf-8 -*-

import sys
import gensim
import sklearn
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import logging
import jieba
jieba.load_userdict("bot/user_disease_hyper_dict/my_hyper_dict.txt")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
TaggededDocument = gensim.models.doc2vec.TaggedDocument

def get_datasest():
    with open("bot/datasets/bot_question.txt", 'r',encoding='utf-8', errors='ignore') as cf:
        docs = cf.readlines()
        print (len(docs))
    x_train = []
    #y = np.concatenate(np.ones(len(docs)))
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l-1] = word_list[l-1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)
    return x_train

def hyper_doc2vec_xtrain(x_train, size=200, epoch_num=1):
    model_dm = Doc2Vec(x_train,min_count=1, window = 3, size = size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('bot/brain/bot_question.doc2vec')
    #return model_dm

def test():
    model_hdd = Doc2Vec.load("bot/brain/bot_question.doc2vec")
    #test_text = ['高血压','者','偶尔','失眠','怎么办','？']
    #test_text = ['高血压者偶尔失眠怎么办？']
    test_text = ['高血压者','偶尔','失眠','怎么办','？']
    inferred_vector_hdd = model_hdd.infer_vector(test_text)
    print (inferred_vector_hdd)
    sims = model_hdd.docvecs.most_similar([inferred_vector_hdd], topn=1)
    return sims

def testx(sentence):
    model_hdd = Doc2Vec.load("bot/brain/bot_question.doc2vec")
    #test_text = ['高血压','者','偶尔','失眠','怎么办','？']
    #test_text = ['高血压者偶尔失眠怎么办？']
    test_text = jieba.lcut(sentence.strip())
    print(test_text)
    inferred_vector_hdd = model_hdd.infer_vector(test_text)
    print (inferred_vector_hdd)
    sims = model_hdd.docvecs.most_similar([inferred_vector_hdd], topn=1)
    return sims

def testxx(sentence):
    model_hdd = Doc2Vec.load("bot/brain/bot_question.doc2vec")
    #test_text = ['高血压','者','偶尔','失眠','怎么办','？']
    #test_text = ['高血压者偶尔失眠怎么办？']
    test_text = list(sentence.strip())
    print(test_text)
    inferred_vector_hdd = model_hdd.infer_vector(test_text)
    print(inferred_vector_hdd)
    sims = model_hdd.docvecs.most_similar([inferred_vector_hdd], topn=1)
    return sims

def hyper_doc2vec_train():
    x_train = get_datasest()
    hyper_doc2vec_xtrain(x_train)

def hyper_doc2vec_predict(sentence):
    #sentence = '高血压者偶尔失眠怎么办？'
    x_train = get_datasest()
    #hyper_doc2vec_xtrain(x_train)
    sims = testxx(sentence)
    for count, sim in sims:
        sentence = x_train[count]
        words = ''
        for word in sentence[0]:
            words = words + word + ' '
        print (words, sim, len(sentence[0]))
        #return words
        res = ''.join(jieba.lcut(words.strip()))
        return res

if __name__ == '__main__':
    x_train = get_datasest()
    #model_dm = train(x_train)
    sentence = '高血压者偶尔失眠怎么办？'
    sims = testxx(sentence)
    for count, sim in sims:
        sentence = x_train[count]
        words = ''
        for word in sentence[0]:
            words = words + word + ' '
        print (words, sim, len(sentence[0]))
    

