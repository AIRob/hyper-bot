from __future__ import unicode_literals, print_function, division
from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from gensim.models import KeyedVectors
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, Activation, RepeatVector, Permute, Lambda, \
    Bidirectional, TimeDistributed, Dropout, Conv1D, GlobalMaxPool1D, Layer
from keras.layers.merge import multiply, concatenate
import keras.backend as K

from keras.preprocessing.sequence import pad_sequences
import numpy as np
import itertools
import jieba

jieba.load_userdict("bot/user_disease_hyper_dict/my_hyper_dict.txt")

# ------------------自定义函数------------------ #

def text_to_word_list(text):  # 文本分词
    text = str(text)
    text = text.split()
    return text

def make_w2v_embeddings(word2vec, df, embedding_dim):  # 将词转化为词向量
    vocabs = {}  # 词序号
    vocabs_cnt = 0  # 词个数计数器

    vocabs_not_w2v = {}  # 无法用词向量表示的词
    vocabs_not_w2v_cnt = 0  # 无法用词向量表示的词个数计数器

    # 停用词
    # stops = set(open('data/stopwords.txt').read().strip().split('\n'))

    for index, row in df.iterrows():
        # 打印处理进度
        if index != 0 and index % 1000 == 0:
            print(str(index) + " sentences embedded.")

        for question in ['question1', 'question2']:
            q2n = []  # q2n -> question to numbers representation
            words = text_to_word_list(row[question])

            for word in words:
                # if word in stops:  # 去停用词
                    # continue
                if word not in word2vec and word not in vocabs_not_w2v:  # OOV的词放入不能用词向量表示的字典中，value为1
                    vocabs_not_w2v_cnt += 1
                    vocabs_not_w2v[word] = 1
                if word not in vocabs:  # 非OOV词，提取出对应的id
                    vocabs_cnt += 1
                    vocabs[word] = vocabs_cnt
                    q2n.append(vocabs_cnt)
                else:
                    q2n.append(vocabs[word])
            df.at[index, question + '_n'] = q2n

    embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)  # 随机初始化一个形状为[全部词个数，词向量维度]的矩阵
    '''
    词1 [a1, a2, a3, ..., a60]
    词2 [b1, b2, b3, ..., b60]
    词3 [c1, c2, c3, ..., c60]
    '''
    embeddings[0] = 0  # 第一行用0填充，因为不存在index为0的词

    for index in vocabs:
        vocab_word = vocabs[index]
        if vocab_word in word2vec:
            embeddings[index] = word2vec[vocab_word]
    del word2vec
    return df, embeddings

def split_and_zero_padding(df, max_seq_length):  # 调整tokens长度

    # 训练集矩阵转换成字典
    X = {'left': df['question1_n'], 'right': df['question2_n']}

    # 调整到规定长度
    for dataset, side in itertools.product([X], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)
    return dataset


class ManDist(Layer):  # 封装成keras层的曼哈顿距离计算

    # 初始化ManDist层，此时不需要任何参数输入
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # 自动建立ManDist层
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # 计算曼哈顿距离
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # 返回结果
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

'''
用于训练孪生网络
'''
# 超参
batch_size = 1024
n_epoch = 9
n_hidden = 50
embedding_dim = 64
max_seq_length = 20

# -----------------基础函数------------------ #
def shared_model(_input,embeddings):
    # 词向量化
    embedded = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_shape=(max_seq_length,),
                         trainable=False)(_input)

    # 多层Bi-LSTM
    activations = Bidirectional(LSTM(n_hidden, return_sequences=True), merge_mode='concat')(embedded)
    activations = Bidirectional(LSTM(n_hidden, return_sequences=True), merge_mode='concat')(activations)

    # dropout
    # activations = Dropout(0.5)(activations)

    # Attention
    attention = TimeDistributed(Dense(1, activation='tanh'))(activations)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(n_hidden * 2)(attention)
    attention = Permute([2, 1])(attention)
    sent_representation = multiply([activations, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    # dropout
    # sent_representation = Dropout(0.1)(sent_representation)
    return sent_representation

def shared_model_cnn(_input,embeddings):
    # 词向量化
    embedded = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_shape=(max_seq_length,),
                         trainable=False)(_input)

    # CNN
    activations = Conv1D(250, kernel_size=5, activation='relu')(embedded)
    activations = GlobalMaxPool1D()(activations)
    activations = Dense(250, activation='relu')(activations)
    activations = Dropout(0.3)(activations)
    activations = Dense(1, activation='sigmoid')(activations)
    return activations

# -----------------训练----------------- #
def siamese_train():
    # ------------------预加载------------------ #
    TRAIN_CSV = 'bot/datasets/hyper_bot_train_data.csv'
    savepath = 'bot/brain/hyperbot_question_siameselstm.h5'
    embedding_dict = {}
    # 读取并加载训练集
    train_df = pd.read_csv(TRAIN_CSV,encoding='gbk')
    for q in ['question1', 'question2']:
        train_df[q + '_n'] = train_df[q]

    # 将训练集词向量化
    train_df, embeddings = make_w2v_embeddings(embedding_dict, train_df, embedding_dim=embedding_dim)
   
    # 分割训练集
    X = train_df[['question1_n', 'question2_n']]
    Y = train_df['is_highly_similar']
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.1)
    X_train = split_and_zero_padding(X_train, max_seq_length)
    X_validation = split_and_zero_padding(X_validation, max_seq_length)

    # 将标签转化为数值
    Y_train = Y_train.values
    Y_validation = Y_validation.values

    # 确认数据准备完毕且正确
    assert X_train['left'].shape == X_train['right'].shape
    assert len(X_train['left']) == len(Y_train)

    left_input = Input(shape=(max_seq_length,), dtype='float32')
    right_input = Input(shape=(max_seq_length,), dtype='float32')
    left_sen_representation = shared_model(left_input,embeddings)
    right_sen_representation = shared_model(right_input,embeddings)

    # 引入曼哈顿距离，把得到的变换concat上原始的向量再通过一个多层的DNN做了下非线性变换、sigmoid得相似度
    # 没有使用https://zhuanlan.zhihu.com/p/31638132中提到的马氏距离，尝试了曼哈顿距离、点乘和cos，效果曼哈顿最好
    man_distance = ManDist()([left_sen_representation, right_sen_representation])
    sen_representation = concatenate([left_sen_representation, right_sen_representation, man_distance])
    similarity = Dense(1, activation='sigmoid')(Dense(2)(Dense(4)(Dense(16)(sen_representation))))
    model = Model(inputs=[left_input, right_input], outputs=[similarity])

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()

    training_start_time = time()
    malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                               batch_size=batch_size, epochs=n_epoch,
                               validation_data=([X_validation['left'], X_validation['right']], Y_validation))
    training_end_time = time()
    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch, training_end_time - training_start_time))

    # Plot accuracy
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.subplot(211)
    plt.plot(malstm_trained.history['acc'])
    plt.plot(malstm_trained.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(212)
    plt.plot(malstm_trained.history['loss'])
    plt.plot(malstm_trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout(h_pad=1.0)
    plt.savefig('bot/imgs/graph.png')

    model.save(savepath)
    print(str(malstm_trained.history['val_acc'][-1])[:6] +
          "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
    print("Finished.")

# -----------------测试----------------- #
def siamese_test():
    # ------------------预加载------------------ #
    savepath = 'bot/brain/hyperbot_question_siameselstm.h5'
    embedding_dict = {}
    # 读取并加载测试集
    TEST_CSV = 'bot/datasets/hyper_bot_test_data.csv'
    test_df = pd.read_csv(TEST_CSV,encoding='gbk')
    for q in ['question1', 'question2']:
        test_df[q + '_n'] = test_df[q]

    # 将测试集词向量化
    test_df, embeddings = make_w2v_embeddings(embedding_dict, test_df, embedding_dim=embedding_dim)
    
    # 预处理
    X_test = split_and_zero_padding(test_df, max_seq_length)
    Y_test = test_df['is_highly_similar'].values

    # 确认数据准备完毕且正确
    assert X_test['left'].shape == X_test['right'].shape
    assert len(X_test['left']) == len(Y_test)

    # 加载预训练好的模型
    model = keras.models.load_model(savepath, custom_objects={"ManDist": ManDist})
    model.summary()
    # 预测并评估准确率
    prediction = model.predict([X_test['left'], X_test['right']])
    print(prediction)
    prediction_list = prediction.tolist()
    accuracy = 0
    for i in range(len(prediction_list)):
        if prediction_list[i][0] < 0.5:
            predict_pro = 0
        else:
            predict_pro = 1
        if predict_pro == Y_test[i]:
            accuracy += 1
    print(accuracy / len(Y_test))

# -----------------预测----------------- #
def siamese_predict():
    savepath = 'bot/brain/hyperbot_question_siameselstm.h5' 
    embedding_dict = {}
    
    # 加载预训练好的词向量和模型
    model = keras.models.load_model(savepath, custom_objects={"ManDist": ManDist})
    model.summary()
    while True:
        # 输入待测试的句对
        sen1 = input("请输入句子1: ")
        sen2 = input("请输入句子2: ")
        dataframe = pd.DataFrame(
            {'question1': [" ".join(jieba.lcut(sen1))], 'question2': [" ".join(jieba.lcut(sen2))]})

        dataframe.to_csv("bot/datasets/test.csv", index=False, sep=',', encoding='utf-8')
        TEST_CSV = 'bot/datasets/test.csv'

        # 读取并加载测试集
        test_df = pd.read_csv(TEST_CSV)
        for q in ['question1', 'question2']:
            test_df[q + '_n'] = test_df[q]

        # 将测试集词向量化
        test_df, embeddings = make_w2v_embeddings(embedding_dict, test_df, embedding_dim=embedding_dim)

        # 预处理
        X_test = split_and_zero_padding(test_df, max_seq_length)

        # 确认数据准备完毕且正确
        assert X_test['left'].shape == X_test['right'].shape

        # 预测并评估准确率
        prediction = model.predict([X_test['left'], X_test['right']])
        print(prediction)

def return_highsim_sentence(sentence):
    # ------------------预加载------------------ #
    savepath = 'bot/brain/hyperbot_question_siameselstm.h5'
    embedding_dict = {}
    with open('bot/datasets/bot_cut_question.txt','r',encoding='utf-8') as data_file:
        data = data_file.readlines()  
    dataframe = pd.DataFrame(
        {'question1': [data[i] for i in range(47592)], 'question2': [" ".join(jieba.lcut(sentence)) for i in range(47592)]})

    dataframe.to_csv("bot/datasets/predict.csv", index=False, sep=',', encoding='utf-8')
    TEST_CSV = 'bot/datasets/predict.csv'

    # 读取并加载测试集
    test_df = pd.read_csv(TEST_CSV)
    for q in ['question1', 'question2']:
        test_df[q + '_n'] = test_df[q]
    # 将测试集词向量化
    test_df, embeddings = make_w2v_embeddings(embedding_dict, test_df, embedding_dim=embedding_dim)
    
    # 预处理
    X_test = split_and_zero_padding(test_df, max_seq_length)
    #Y_test = test_df['is_highly_similar'].values

    # 确认数据准备完毕且正确
    assert X_test['left'].shape == X_test['right'].shape
    #assert len(X_test['left']) == len(Y_test)

    # 加载预训练好的模型
    model = keras.models.load_model(savepath, custom_objects={"ManDist": ManDist})
    model.summary()
    # 预测并评估准确率
    prediction = model.predict([X_test['left'], X_test['right']])
    #prediction = as_num(prediction)
    #print(prediction)
    prediction_list = prediction.tolist()
    max_pred = 0.1
    indx = 0
    for i in range(len(prediction_list)):
        curr_pred = float(as_num(float(prediction_list[i][0])))
        #if curr_pred > 0.90 and curr_pred < 1.0:
        if curr_pred > max_pred and curr_pred < 0.99:
            max_pred = curr_pred
            indx = i
    
    sim_res = ''.join(test_df.iloc[indx]['question1'].split(" "))
    #return sim_res
    res = ''.join(jieba.lcut(sim_res.strip()))
    return res

def as_num(x):
    y='{:.5f}'.format(x) # 5f表示保留5位小数点的float型
    return y

if __name__ == '__main__':
    #siamese_train()
    #siamese_test()
    #siamese_predict()
    #sen2 = '高血压失眠怎么办？'
    sen2 = '高血压与失眠有关?吗？'
    print(return_highsim_sentence(sen2))
