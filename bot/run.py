from __future__ import unicode_literals, print_function, division
import sys
import json
import random
from bot.secret_weapon.action_models import action_train, action_predict
#from bot.secret_weapon.seqtoseq_model import trainIters, reply_predict
#from bot.secret_weapon.seq2seq_models import trainIters, Lang,readLangs,EncoderRNN,AttnDecoderRNN,predictReply
#from bot.secret_weapon.bot_seq2seq import bot_train_pre,predictReplyX
from bot.secret_weapon.bot_seq2seq import pre_train_bot,predictReplyX,bot_predict_reply,bot_predict_sim_reply
from bot.secret_weapon.question_siamese import return_highsim_sentence,siamese_train
from bot.secret_weapon.sim_sentence_api import simhash_x,jaro_winkler_x
from bot.secret_weapon.bot_doc2vec_models import doc2vec_sim_reply,doc2vec_train
from bot.knowledge_graph.BotKGInferMain import botKGInferAllLabel
from bot.secret_weapon.hyper_doc2vec_models import hyper_doc2vec_train,hyper_doc2vec_predict
from bot.secret_weapon.lsi_sentence_sim import lsi_sim_sentence,lsi_model
import jieba
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu" if torch.cuda.is_available() else "cuda")

jieba.load_userdict("bot/user_disease_hyper_dict/my_hyper_dict.txt")

def log(message):
    """log function"""
    if message:
        print(str(message))
        sys.stdout.flush()
    else:
        print("NULL")
        sys.stdout.flush()

def run_bot(sentence):
    """function to run the bot"""
    #sentence = ' '.join(jieba.lcut(sentence.strip()))
    print(sentence)
    intent = action_predict(str(sentence))
    #log(intent)
    reply = dsl_protocol(intent, sentence)
    #test
    reply = botKGInferAllLabel(sentence)
    if reply == "none":
        #log(reply)
        #reply = predictReply(str(sentence))
        #reply = replySentenceXX(str(sentence))
        #reply = predictReplyX(str(sentence))
        reply = bot_predict_reply(str(sentence))
        print(reply)
        if reply == "error":
            #reply = bot_predict_sim_reply(str(sentence))
            #sim_sentence = return_highsim_sentence(str(sentence))
            #sim_sentence = simhash_x(sentence)
            #reply = bot_predict_reply(str(sim_sentence))
            #sim_sentence = doc2vec_sim_reply(sentence)
            #sim_sentence = hyper_doc2vec_predict(sentence)
            #print(sim_sentence)
            #reply = bot_predict_reply(str(sim_sentence))
            # return_highsim_sentence function return 
            sent_len = len(str(sentence).strip())
            if sent_len < 9:sim_sentence = jaro_winkler_x(str(sentence))
            elif sent_len > 8 and sent_len < 21:sim_sentence = lsi_sim_sentence(str(sentence))
            elif sent_len > 20 and sent_len < 41:sim_sentence = return_highsim_sentence(str(sentence))
            elif sent_len > 40 and sent_len < 200:sim_sentence = hyper_doc2vec_predict(str(sentence))
            else:sim_sentence = simhash_x(str(sentence))
            
            print(sim_sentence)
            reply = bot_predict_reply(str(sim_sentence))
            if reply == "error":
                reply = "Sorry, I can not understand this topic very well, can you change your topic?"
    return reply

def run_bot_pre():
    """function to run the bot""" 
    action_train_pre()
    
    #doc2vec_train()
    siamese_train()
    hyper_doc2vec_train()
    lsi_model()
    pre_train_bot()
 
def run_action_bot_pre():
    """function to run the action bot"""   
    action_train_pre()
    hyper_doc2vec_train()

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    #pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def seq_train_pre():
    """function to train the seqtoseq model"""
    print("Reading lines...")
    input_lang, output_lang, pairs = prepareData('quesn', 'ansr', False)
    print(random.choice(pairs))
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    #trainIters(pairs,75000,tfl=False)
    #trainIters(10000, training_data, tfl=True)
    trainIters(encoder1, attn_decoder1, 10000, print_every=1000,tfl=False)

def seq_train_protocol(sentence):
    """function to train the seqtoseq model"""
    print("Reading lines...")
    input_lang, output_lang, pairs = prepareData('quesn', 'ansr', False)
    print(random.choice(pairs))
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    #trainIters(pairs,75000,tfl=False)
    #trainIters(10000, training_data, tfl=True)
    trainIters(encoder1, attn_decoder1, 10000, print_every=1000,tfl=False)
    print(predictReply(sentence))


def bot_seq_train_pre():
    """function to train the seq2seq prediction model"""
    corpus_path = 'bot/datasets/hyperqas.txt'
    trainIters(corpus_path, True, 10000, 0.0001, 64,
                    1, 512, 100, 1000, 0.1)
    #print(predictReplyX(sentence))
    #print(replySentenceXX(sentence))

def action_train_pre():
    """function to train the action prediction model"""
    training_data = []
    with open('bot/datasets/action_dataset.json') as data_file:
        data = json.load(data_file)
    for line in data:
        #fetching training data
        training_data.append(line)
    action_train(10000, training_data) #training the model

def action_train_protocol(sentence):
    """function to train the action prediction model"""
    training_data = []
    with open('bot/datasets/action_dataset.json') as data_file:
        data = json.load(data_file)

    for line in data:
        #fetching training data
        training_data.append(line)

    action_train(10000, training_data) #training the model

    print("intent:" + action_predict(sentence))

def test_run_protocol():
    """function for test running the bot"""
    while True:
        k = input("user: ")
        print("hybot: ", run_bot(k))

def dsl_protocol(intent, sentence):
    """domain specific language for the bot"""
    rep = {}
    rep["text"] = "none"
    key = {'contact':'You can send email to this email id hr@gaojihealth.com', 'baike':'https://baike.baidu.com/item/高济/47725', 'phone':'010 - 56939988'
       , 'open':'http://www.gaojimed.com/', 'location':'北京市东城区环球贸易中心B座16层'}
    if intent != "none":
        rep["text"] = random.choice(["check this out:", "here you go:", "I found this:"]) + key[intent]
    return rep["text"]

