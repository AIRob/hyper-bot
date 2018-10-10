from __future__ import unicode_literals, print_function, division
import re
import os
import unicodedata
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.backends.cudnn as cudnn

import itertools
import random
import math
from tqdm import tqdm
#from bot.secret_weapon.question_siamese import return_highsim_sentence


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
#device = torch.device("cpu" if USE_CUDA else "cuda")
MAX_LENGTH = 200
teacher_forcing_ratio = 1.0
SAVE_DIR = 'bot/brain'
SOS_token = 0
EOS_token = 1
PAD_token = 2


######################################################################
# 数据预处理
# ==================
#
# 每个单词没法直接作为网络的输入，
# 需要给每个不同的单词一个下标作为它的标签 
#
class Lang:
    '''
    SOS表示一句话的开始, EOS表示一句话的结束, PAD表示填充
    '''
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"PAD"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
#通过上面代码将Unicode的字符转化成ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readLangs(corpus, corpus_name):
    '''
    数据集是一对一对的放在txt里，定义ReadLang函数读取内容并保存
    '''
    print("Reading lines...")

    # combine every two lines into pairs and normalize
    with open(corpus) as f:
        content = f.readlines()
    # import gzip
    # content = gzip.open(corpus, 'rt')
    lines = [x.strip() for x in content]
    it = iter(lines)
    # pairs = [[normalizeString(x), normalizeString(next(it))] for x in it]
    pairs = [[x, next(it)] for x in it]

    voc = Lang(corpus_name)
    return voc, pairs

#过滤掉长度小于MAX_LENGTH的句子作为训练集进行训练
def filterPair(p):
    # input sequences need to preserve the last word for EOS_token
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

#拼接前面定义的函数构成新的函数
def prepareData(corpus, corpus_name):
    '''
    1、读取数据readLangs
    2、过滤不满足条件的数据filterPairs
    3、将qa对放在同一字典下，将所有出现过的词语一个标签对应起来addSentence 
    '''
    voc, pairs = readLangs(corpus, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.n_words)
    directory = os.path.join(SAVE_DIR, 'training_data', corpus_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(voc, os.path.join(directory, '{!s}.tar'.format('voc')))
    torch.save(pairs, os.path.join(directory, '{!s}.tar'.format('pairs')))
    return voc, pairs

def loadPrepareData(corpus):
    corpus_name = corpus.split('/')[-1].split('.')[0]
    try:
        print("Start loading training data ...")
        voc = torch.load(os.path.join(SAVE_DIR, 'training_data', corpus_name, 'voc.tar'))
        pairs = torch.load(os.path.join(SAVE_DIR, 'training_data', corpus_name, 'pairs.tar'))
    except FileNotFoundError:
        print("Saved data not found, start preparing trianing data ...")
        voc, pairs = prepareData(corpus, corpus_name)
    return voc, pairs

######################################################################
# 定义好编码器
# ==================

class EncoderRNN(nn.Module):
    '''

    '''
    def __init__(self, input_size, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden) # output: (seq_len, batch, hidden*n_dir)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:] # Sum bidirectional outputs (1, batch, hidden)
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden [1, 64, 512], encoder_outputs [14, 64, 512]
        max_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)

        # Create variable to store attention energies
        attn_energies = torch.zeros(batch_size, max_len) # B x S
        attn_energies = attn_energies.to(device)

        # For each batch of encoder outputs
        for b in range(batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_output):
        # hidden [1, 512], encoder_output [1, 512]
        if self.method == 'dot':
            energy = hidden.squeeze(0).dot(encoder_output.squeeze(0))
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.squeeze(0).dot(energy.squeeze(0))
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.squeeze(0).dot(energy.squeeze(0))
            return energy

######################################################################
# 带有注意力机制的解码器
# ==================
#
class AttnDecoderRNN(nn.Module):
    '''
    注意力机制简要阐述
    首先将网络输入转化为词向量
    然后将词向量与隐藏状态拼接在一起
    最后通过softmax激活层输出固定长度的序列
    '''
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        '''
        
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        '''
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded) #[1, 64, 512]
        if(embedded.size(0) != 1):
            raise ValueError('Decoder input sequence length should be 1')

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs) #[64, 1, 14]
        # encoder_outputs [14, 64, 512]
        #通过批量乘法输出结果
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) #[64, 1, 512]

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) #[64, 512]
        context = context.squeeze(1) #[64, 512]
        #将bmm结果与rnn_output拼接一起
        concat_input = torch.cat((rnn_output, context), 1) #[64, 1024]
        concat_output = F.tanh(self.concat(concat_input)) #[64, 512]

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output) #[64, output_size]
        output = F.softmax(output, dim=1)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

######################################################################
# 基于attention机制的seq2seq模型训练
# ==================
#

cudnn.benchmark = True
#############################################
# generate file name for saving parameters
#############################################
def filename(reverse, obj):
	filename = ''
	if reverse:
		filename += 'reverse_'
	filename += obj
	return filename


#############################################
# Prepare Training Data
#############################################
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

# batch_first: true -> false, i.e. shape: seq_len * batch
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# convert to index, add EOS
# return input pack_padded_sequence
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = [len(indexes) for indexes in indexes_batch]
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# convert to index, add EOS, zero padding
# return output variable, mask, max length of the sentences in batch
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# pair_batch is a list of (input, output) with length batch_size
# sort list of (input, output) pairs by input length, reverse input
# return input, lengths for pack_padded_sequence, output_variable, mask
def batch2TrainData(voc, pair_batch, reverse):
    if reverse:
        pair_batch = [pair[::-1] for pair in pair_batch]
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

#############################################
# Training
#############################################

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, max_length=MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    loss = 0
    print_losses = []
    n_totals = 0

    encoder_outputs, encoder_hidden = encoder(input_variable, lengths, None)

    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Run through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_input = target_variable[t].view(1, -1) # Next input is current target
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal) #v0.4
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            _, topi = decoder_output.topk(1) # [64, 1]

            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    loss.backward()

    clip = 50.0
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(corpus, reverse, n_iteration, learning_rate, batch_size, n_layers, hidden_size,
                print_every, save_every, dropout, loadFilename=None, attn_model='dot', decoder_learning_ratio=5.0):

    voc, pairs = loadPrepareData(corpus)

    # training data
    corpus_name = os.path.split(corpus)[-1].split('.')[0]
    training_batches = None
    try:
        training_batches = torch.load(os.path.join(SAVE_DIR, 'training_data', corpus_name,
                                                   '{}_{}_{}.tar'.format(n_iteration, \
                                                                         filename(reverse, 'training_batches'), \
                                                                         batch_size)))
    except FileNotFoundError:
        print('Training pairs not found, generating ...')
        training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)], reverse)
                          for _ in range(n_iteration)]
        torch.save(training_batches, os.path.join(SAVE_DIR, 'training_data', corpus_name,
                                                  '{}_{}_{}.tar'.format(n_iteration, \
                                                                        filename(reverse, 'training_batches'), \
                                                                        batch_size)))
    # model
    checkpoint = None
    print('Building encoder and decoder ...')
    embedding = nn.Embedding(voc.n_words, hidden_size)
    encoder = EncoderRNN(voc.n_words, hidden_size, embedding, n_layers, dropout)
    attn_model = 'dot'
    decoder = AttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers, dropout)
    if loadFilename:
        checkpoint = torch.load(loadFilename)
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])
    # use cuda
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # optimizer
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])

    # initialize
    print('Initializing ...')
    start_iteration = 1
    perplexity = []
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1
        perplexity = checkpoint['plt']

    for iteration in tqdm(range(start_iteration, n_iteration + 1)):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size)
        print_loss += loss
        perplexity.append(loss)

        if iteration % print_every == 0:
            print_loss_avg = math.exp(print_loss / print_every)
            print('%d %d%% %.4f' % (iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        if (iteration % save_every == 0):
            directory = os.path.join(SAVE_DIR, 'model', corpus_name, '{}-{}_{}'.format(n_layers, n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'plt': perplexity
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, filename(reverse, 'backup_bidir_model'))))

######################################################################
# 模型评估
# ==================
#

class Sentence:
    def __init__(self, decoder_hidden, last_idx=SOS_token, sentence_idxes=[], sentence_scores=[]):
        if(len(sentence_idxes) != len(sentence_scores)):
            raise ValueError("length of indexes and scores should be the same")
        self.decoder_hidden = decoder_hidden
        self.last_idx = last_idx
        self.sentence_idxes =  sentence_idxes
        self.sentence_scores = sentence_scores

    def avgScore(self):
        if len(self.sentence_scores) == 0:
            raise ValueError("Calculate average score of sentence, but got no word")
        # return mean of sentence_score
        return sum(self.sentence_scores) / len(self.sentence_scores)

    def addTopk(self, topi, topv, decoder_hidden, beam_size, voc):
        topv = torch.log(topv)
        terminates, sentences = [], []
        for i in range(beam_size):
            if topi[0][i] == EOS_token:
                terminates.append(([voc.index2word[idx.item()] for idx in self.sentence_idxes] + ['<EOS>'],
                                   self.avgScore())) # tuple(word_list, score_float
                continue
            idxes = self.sentence_idxes[:] # pass by value
            scores = self.sentence_scores[:] # pass by value
            idxes.append(topi[0][i])
            scores.append(topv[0][i])
            sentences.append(Sentence(decoder_hidden, topi[0][i], idxes, scores))
        return terminates, sentences

    def toWordScore(self, voc):
        words = []
        for i in range(len(self.sentence_idxes)):
            if self.sentence_idxes[i] == EOS_token:
                words.append('<EOS>')
            else:
                words.append(voc.index2word[self.sentence_idxes[i].item()])
        if self.sentence_idxes[-1] != EOS_token:
            words.append('<EOS>')
        return (words, self.avgScore())

def beam_decode(decoder, decoder_hidden, encoder_outputs, voc, beam_size, max_length=MAX_LENGTH):
    terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
    prev_top_sentences.append(Sentence(decoder_hidden))
    for i in range(max_length):
        for sentence in prev_top_sentences:
            decoder_input = torch.LongTensor([[sentence.last_idx]])
            decoder_input = decoder_input.to(device)

            decoder_hidden = sentence.decoder_hidden
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.topk(beam_size)
            term, top = sentence.addTopk(topi, topv, decoder_hidden, beam_size, voc)
            terminal_sentences.extend(term)
            next_top_sentences.extend(top)

        next_top_sentences.sort(key=lambda s: s.avgScore(), reverse=True)
        prev_top_sentences = next_top_sentences[:beam_size]
        next_top_sentences = []

    terminal_sentences += [sentence.toWordScore(voc) for sentence in prev_top_sentences]
    terminal_sentences.sort(key=lambda x: x[1], reverse=True)

    n = min(len(terminal_sentences), 15)
    return terminal_sentences[:n]

def decode(decoder, decoder_hidden, encoder_outputs, voc, max_length=MAX_LENGTH):

    decoder_input = torch.LongTensor([[SOS_token]])
    decoder_input = decoder_input.to(device)

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length) #TODO: or (MAX_LEN+1, MAX_LEN+1)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        _, topi = decoder_output.topk(3)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(voc.index2word[ni.item()])

        decoder_input = torch.LongTensor([[ni]])
        decoder_input = decoder_input.to(device)

    return decoded_words, decoder_attentions[:di + 1]


def evaluate(encoder, decoder, voc, sentence, beam_size, max_length=MAX_LENGTH):
    indexes_batch = [indexesFromSentence(voc, sentence)] #[1, seq_len]
    lengths = [len(indexes) for indexes in indexes_batch]
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)

    encoder_outputs, encoder_hidden = encoder(input_batch, lengths, None)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    if beam_size == 1:
        return decode(decoder, decoder_hidden, encoder_outputs, voc)
    else:
        return beam_decode(decoder, decoder_hidden, encoder_outputs, voc, beam_size)


def evaluateRandomly(encoder, decoder, voc, pairs, reverse, beam_size, n=10):
    for _ in range(n):
        pair = random.choice(pairs)
        print("=============================================================")
        if reverse:
            print('>', " ".join(reversed(pair[0].split())))
        else:
            print('>', pair[0])
        if beam_size == 1:
            output_words, _ = evaluate(encoder, decoder, voc, pair[0], beam_size)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
        else:
            output_words_list = evaluate(encoder, decoder, voc, pair[0], beam_size)
            for output_words, score in output_words_list:
                output_sentence = ' '.join(output_words)
                print("{:.3f} < {}".format(score, output_sentence))

def evaluateInput(encoder, decoder, voc, beam_size):
    pair = ''
    while(1):
        try:
            pair = input('> ')
            if pair == 'q': break
            if beam_size == 1:
                output_words, _ = evaluate(encoder, decoder, voc, pair, beam_size)
                output_sentence = ' '.join(output_words)
                print('<', output_sentence)
            else:
                output_words_list = evaluate(encoder, decoder, voc, pair, beam_size)
                for output_words, score in output_words_list:
                    output_sentence = ' '.join(output_words)
                    print("{:.3f} < {}".format(score, output_sentence))
        except KeyError:
            return ("Incorrect spelling.")


def runTest(n_layers, hidden_size, reverse, modelFile, beam_size, inp, corpus):
    torch.set_grad_enabled(False)

    voc, pairs = loadPrepareData(corpus)
    embedding = nn.Embedding(voc.n_words, hidden_size)
    encoder = EncoderRNN(voc.n_words, hidden_size, embedding, n_layers)
    attn_model = 'dot'
    decoder = AttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers)

    checkpoint = torch.load(modelFile)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    # train mode set to false, effect only on dropout, batchNorm
    encoder.train(False);
    decoder.train(False);

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    if inp:
        evaluateInput(encoder, decoder, voc, beam_size)
    else:
        evaluateRandomly(encoder, decoder, voc, pairs, reverse, beam_size, 20)

######################################################################
# 准备数据 and  利用模型
# ==================
#
def parseFilename(filename, test=False):
    filename = filename.split('/')
    dataType = filename[-1][:-4] # remove '.tar'
    parse = dataType.split('_')
    reverse = 'reverse' in parse
    layers, hidden = filename[-2].split('_')
    n_layers = int(layers.split('-')[0])
    hidden_size = int(hidden)
    return n_layers, hidden_size, reverse

def pre_train_bot():
    corpus_path = 'bot/datasets/hyperqas.txt'
    model_path = 'bot/brain/model/hyperqas/1-1_512/10000_backup_bidir_model.tar'
    if corpus_path and not model_path:
        trainIters(corpus_path, True, 10000, 0.0001, 64,
                    1, 512, 100, 1000, 0.1)
    elif model_path:
        n_layers, hidden_size, reverse = parseFilename(model_path)
        trainIters(corpus_path, reverse, 10000, 0.0001, 64,
                    n_layers, hidden_size, 100, 1000, 0.1, loadFilename=None)
    elif model_path and corpus_path:
        n_layers, hidden_size, reverse = parseFilename(model_path, True)
        runTest(n_layers, hidden_size, reverse, model_path, 1, True,corpus_path )


def predictReply(pair,encoder, decoder, voc, beam_size):
    #pair = '高血压心脏病喝什么茶最好'
    while(1):
        try:
            #pair = input('> ')
            if pair == 'q': break
            if beam_size == 1:
                output_words, _ = evaluate(encoder, decoder, voc, pair, beam_size)
                output_sentence = ' '.join(output_words)
                print("< {}".format(output_sentence))
                #return output_sentence.split("<EOS")[0]
                return output_sentence.split("<EOS")[0]
            else:
                output_words_list = evaluate(encoder, decoder, voc, pair, beam_size)
                for output_words, score in output_words_list:
                    output_sentence = ' '.join(output_words)
                    print("{:.3f} < {}".format(score, output_sentence))
                    return output_sentence.split("<EOS")[0]
        except KeyError:
            return ("error")

def predictReplyX(pair):
    corpus_path = 'bot/datasets/hyperqas.txt'
    model_path = 'bot/brain/model/hyperqas/1-1_512/10000_backup_bidir_model.tar'
    #if corpus_path and not model_path:
    #    trainIters(corpus_path, True, 1000, 0.0001, 64,1, 512, 100, 1000, 0.1)
    '''
    elif model_path:
        n_layers, hidden_size, reverse = parseFilename(model_path)
        trainIters(corpus_path, reverse, 10000, 0.0001, 64,
                    n_layers, hidden_size, 100, 1000, 0.1, loadFilename=model_path)
    '''
    if model_path:
        n_layers, hidden_size, reverse = parseFilename(model_path, True)
        return runTestX(pair,n_layers, hidden_size, reverse, model_path, 1, True, corpus_path)

def runTestX(pair,n_layers, hidden_size, reverse, modelFile, beam_size, inp, corpus):
    torch.set_grad_enabled(False)

    voc, pairs = loadPrepareData(corpus)
    embedding = nn.Embedding(voc.n_words, hidden_size)
    encoder = EncoderRNN(voc.n_words, hidden_size, embedding, n_layers)
    attn_model = 'dot'
    decoder = AttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers)

    checkpoint = torch.load(modelFile)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    # train mode set to false, effect only on dropout, batchNorm
    encoder.train(False);
    decoder.train(False);

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    if inp:
        return predictReply(pair,encoder, decoder, voc, beam_size)
    else:
        evaluateRandomly(encoder, decoder, voc, pairs, reverse, beam_size, 20)

def bot_predict_reply(sentence):
    modelFile = 'bot/brain/model/hyperqas/1-1_512/10000_backup_bidir_model.tar'
    corpus = 'bot/datasets/hyperqas.txt'
    n_layers, hidden_size, reverse = parseFilename(modelFile, True)
    try:
        torch.set_grad_enabled(False)
  
        voc, pairs = loadPrepareData(corpus)

        embedding = nn.Embedding(voc.n_words, hidden_size)
        encoder = EncoderRNN(voc.n_words, hidden_size, embedding, n_layers)
        attn_model = 'dot'
        decoder = AttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers)

        checkpoint = torch.load(modelFile)
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])

        # train mode set to false, effect only on dropout, batchNorm
        encoder.train(False);
        decoder.train(False);

        encoder = encoder.to(device)
        decoder = decoder.to(device)
        
        output_words, attentions = evaluate(encoder, decoder,voc,sentence,1)
        output_sentence = ' '.join(output_words)
        return output_sentence.split("<EOS>")[0]
        
    except(KeyError):
        return "error"

def bot_predict_sim_reply(sentence):
    sim_sentence = return_highsim_sentence(sentence)
    print(sim_sentence)
    modelFile = 'bot/brain/model/hyperqas/1-1_512/10000_backup_bidir_model.tar'
    corpus = 'bot/datasets/hyperqas.txt'
    n_layers, hidden_size, reverse = parseFilename(modelFile, True)
    try:
        torch.set_grad_enabled(False)
  
        voc, pairs = loadPrepareData(corpus)

        embedding = nn.Embedding(voc.n_words, hidden_size)
        encoder = EncoderRNN(voc.n_words, hidden_size, embedding, n_layers)
        attn_model = 'dot'
        decoder = AttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers)

        checkpoint = torch.load(modelFile)
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])

        # train mode set to false, effect only on dropout, batchNorm
        encoder.train(False);
        decoder.train(False);

        encoder = encoder.to(device)
        decoder = decoder.to(device)
        
        output_words, attentions = evaluate(encoder, decoder,voc,sim_sentence,1)
        output_sentence = ' '.join(output_words)
        return output_sentence.split("<EOS>")[0]
        
    except(KeyError):
        return "error"

if __name__ == '__main__':
    pre_train_bot()
    sentence = '高血压心脏病喝什么茶最好'
    print(predictReplyX(sentence))
    print(bot_predict_reply(sentence))
