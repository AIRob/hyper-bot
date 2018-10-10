from __future__ import unicode_literals, print_function, division
import pickle as pk
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import json
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu" if torch.cuda.is_available() else "cuda")

class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size)
        #self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inpt):

        # forward pass of the network
        hidden = self.i2h(inpt)   # input to hidden layer
        #hidden = self.h2h(hidden)
        output = self.h2o(hidden)  # hidden to output layer
        output = self.softmax(output)   # softmax layer
        return output


def dataclean(training_data):

    all_categories = list()  # list to store categories of intent
    all_words = list()  # for storing all words to convert input sentence into bag of words
    for data in training_data:
        if data["intent"] not in all_categories:
            all_categories.append(data["intent"])
        for word in data["sentence"].split(" "):
            #  storing words in each sentence
            if word not in all_words:
                all_words.append(word)
    return all_categories, all_words

def wordToIndex(word,all_words):
    # finding indx of a word from all_words
    return all_words.index(word)

def sentencetotensor(sentence,all_words):
    # input tensor initialized with zeros
    n_words = len(all_words)
    tensor = torch.zeros(1, n_words)
    for word in sentence.split(" "):
        if word not in all_words:
            # to deal with words not in dataset in evaluation stage
            continue
        tensor[0][wordToIndex(word, all_words)] = 1  # making found word's position 1
    return tensor

def randomchoice(length):
    # random function for shuffling dataset
    return length[random.randint(0, len(length)-1)]

def randomtrainingexample(training_data,all_categories,all_words):
    # produce random training data
    data = randomchoice(training_data)
    category = data['intent']
    category_tensor = torch.tensor([all_categories.index(category)],device=device)
    # creating target Tensor
    sentence = data["sentence"]  # input
    line_tensor = sentencetotensor(sentence, all_words)  # input tensor
    return sentence, category_tensor, line_tensor

def train(output, input, ann,learning_rate=.005):
    # function for training the neural net
    criterion = nn.NLLLoss()
    ann.zero_grad()  # initializing gradients with zeros

    # predicting the output
    output_p = ann(input)  # input --> hidden_layer --> output
    loss = criterion(output_p, output)
    # comparing the guessed output with actual output
    loss.backward()  # backpropagating to compute gradients with respect to loss

    for p in ann.parameters():
        # adding learning rate to slow down the network
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.item()  # returning predicted output and loss

#n_iters=100000
def action_train(n_iters, training_data):
    all_categories, all_words = dataclean(training_data)
    with open('bot/brain/action_meta.pkl','rb') as pickle_file:
         meta = pk.load(pickle_file)  

    #off = meta[2]
    current_loss = 0
    input_size = len(all_words)
    output_size = len(all_categories)
    hidden_size = 128
    ann = ANN(input_size, hidden_size, output_size)  # will initialize the computation graph

    for iter in range(1, n_iters+1):
        # training the network for n_iteration
        sentence, category_tensor, line_tensor = randomtrainingexample(training_data, all_categories, all_words)
        # fetching random training data
        output, loss = train(category_tensor, line_tensor, ann)
        # training the neural network to predict the intent accuratly
        current_loss += loss  # updating the error
            # for each 50 iteration print the error,input,actual intent,guessed intent
           #top_v,top_i=output.data.topk(1)
        k = 0
        output_index = output.data.numpy()[0]
        top_v, _ = output.data.topk(1)

            # converting output tensor to integer
        out_index = category_tensor.data.numpy()  # converting tensor datatype to integer
        accuracy = 100-(loss*100)
        if accuracy < 0:
            accuracy = 0

        if iter%100 == 0:
           print('accuracy=', round(accuracy), '%', 'input=', sentence, 'actual=', all_categories[out_index[0]],'guess=', all_categories[output_index])
    
    with open('bot/brain/action_meta.pkl','rb') as pickle_file:
         meta = pk.load(pickle_file)
    
    #thresh = meta[2]
    thresh = -1
    data = ["hello", "how are you", "are you ok","this is awesome","do you love me","will you marry me","blah blah blah","duck", "fuck you", "you are beautiful", "i love you", "tell me a joke", "are you mad", "i know you are sad","everything will be ok"]
    for sentence in data:
        output = evaluate(sentencetotensor(sentence, all_words), ann)
        top_v, top_i = output.data.topk(1)
        if top_v[0][0] > thresh:
           thresh = top_v[0][0]
    torch.save(ann, 'bot/brain/ann.pth')  
    metadata = open('bot/brain/action_meta.pkl', 'wb')
    pk.dump([all_categories, all_words,thresh], metadata)
    
def evaluate(line_tensor, ann):

    # output evaluating function
    output = ann(line_tensor)
    return output

def action_predict(sentence):
    ann = torch.load('bot/brain/ann.pth')
    with open('bot/brain/action_meta.pkl','rb') as pickle_file:
         meta = pk.load(pickle_file)
    all_categories = meta[0]
    all_words = meta[1]
    thresh = meta[2]
    # function for evaluating user input sentence
    # print("input=",sentence)
    output = evaluate(sentencetotensor(sentence, all_words), ann)
    top_v, top_i = output.data.topk(1)
    #print(top_v[0][0])
    output_index = top_i[0][0]
    if top_v[0][0]>thresh:
       return all_categories[output_index]
    else:
       return "none"
