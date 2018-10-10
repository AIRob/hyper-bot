from __future__ import unicode_literals, print_function, division
from simhash import Simhash 
import numpy as np 
import math
import Levenshtein as ls
import pandas as pd 
#from bot.secret_weapon.bot_seq2seq import bot_predict_reply


class SimWord(object):
    """docstring for SimWord"""
    def __init__(self):
        pass
        
    def simhash_func(self,str1,str2):
        '''
        simhash计算唯一性
        '''
        res =  Simhash(str1).distance(Simhash(str2))
        return res

    def numsim_func(self,num1,num2):
        '''
        数值大小唯一性识别，比如0.87与0.88是非常接近的
        '''
        mid = (num1 + num2)/2.0
        numres = np.sqrt((math.pow(num1-mid,2)+math.pow(num2-mid,2)) / 2.0) / mid
        return numres

    def jaro_winkler_func(self,str1,str2):
        '''
        编辑距离jaro_winkler算法
        参考https://www.cnblogs.com/zangrunqiang/p/6752430.html
        针对多字错字错位的情况相似度特高测试代码dis_test.py效果挺好
        '''
        res = ls.jaro_winkler(str1,str2)
        return res

    def dislocation_func(self,str1,str2):
        '''
        编辑距离jaro_winkler算法
        参考https://www.cnblogs.com/zangrunqiang/p/6752430.html
        针对多字错字错位的情况相似度效果不佳
        '''   
        res = ls.ratio(str1,str2)
        return res

    def distance_func(self,str1,str2):
        '''
        编辑距离jaro_winkler算法
        参考https://www.cnblogs.com/zangrunqiang/p/6752430.html
        针对多字错字错位的情况相似度效果不佳
        '''
        res = ls.distance(str1,str2)
        return res

def read_datas():
    with open('bot/datasets/bot_question.txt','r',encoding='utf-8') as f:
        data = f.readlines() 
    return data

def simhash_x(sentence):
    data = read_datas()
    sim_list = []
    for i in range(47592):
        str1 = data[i].strip()
        sd = SimWord()
        jwres = sd.simhash_func(str1,sentence)
        sim_list.append(jwres)
    indx = np.argmin(sim_list)
    for j in range(47592):
        res = data[j].strip()
        if j == indx:
            print (f'{res}:{sim_list[indx]}')
            return res
            '''
            if res is not None:return res
            else:return sentence
            '''

def simhash_reply(sentence):

    '''
    data = read_datas()
    min_dis = 0.1
    sim_list = []
    for i in range(47592):
        str1 = data[i].strip()
        sd = SimWord()
        jwres = sd.simhash_func(str1,sentence)
        sim_list.append(jwres)
    indx = np.argmin(sim_list)
    for j in range(47592):
        res = data[j].strip()
        if j == indx:
            print (f'{res}:{sim_list[indx]}')
            #return res
            print(bot_predict_reply(res))
            return bot_predict_reply(res)
        else:
            return 'error'
    '''
    return sentence
    
def main():
    str1 = '高血压，动脉硬化失眠.，怎么办？'
    str2 = '高血压，动脉硬化失眠，怎么办？'
    str3 = '朋友是高血压引发的肾病患者，想知道高血压导致的肾病的患者需要怎么饮食？'
    str4 = '朋友是高血压引发的肾病患者，想知道高血压引发的肾病患者的危害有哪些呢？'
    str5 = '失眠会引起高血压吗'
    str6 = '高血压者偶尔失眠怎么办？'
    str7 = '高血压心脏病能做飞机吗'
    str8 = '高血压心脏病能坐飞机吗'
    simhash_x(str1)
    simhash_x(str2)
    simhash_x(str5)
    
if __name__ == '__main__':
    main()
