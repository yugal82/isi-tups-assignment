from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import json
import gzip
import torch
import numpy as np
import os.path
import re
import torch.nn.functional as F
import itertools
from nltk.tokenize import word_tokenize
import torch.nn as nn
import tqdm
import pandas as pd
import re
from functools import partial
from multiprocessing import Pool

REG = False
NUM_THREADS = 2


# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
This file contains the definition of encoders used in https://arxiv.org/pdf/1705.02364.pdf
"""

import numpy as np
import time

import torch
import torch.nn as nn


class InferSent(nn.Module):

    def __init__(self, config):
        super(InferSent, self).__init__()
        self.bsize = config['bsize']
        self.word_emb_dim = config['word_emb_dim']
        self.enc_lstm_dim = config['enc_lstm_dim']
        self.pool_type = config['pool_type']
        self.dpout_model = config['dpout_model']
        self.version = 1 if 'version' not in config else config['version']

        self.enc_lstm = nn.LSTM(self.word_emb_dim, self.enc_lstm_dim, 1,
                                bidirectional=True, dropout=self.dpout_model)

        assert self.version in [1, 2]
        if self.version == 1:
            self.bos = '<s>'
            self.eos = '</s>'
            self.max_pad = True
            self.moses_tok = False
        elif self.version == 2:
            self.bos = '<p>'
            self.eos = '</p>'
            self.max_pad = False
            self.moses_tok = True

    def is_cuda(self):
        # either all weights are on cpu or they are on gpu
        return self.enc_lstm.bias_hh_l0.data.is_cuda

    def forward(self, sent_tuple):
        # sent_len: [max_len, ..., min_len] (bsize)
        # sent: (seqlen x bsize x worddim)
        sent, sent_len = sent_tuple

        # Sort by length (keep idx)
        sent_len_sorted, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
        sent_len_sorted = sent_len_sorted.copy()
        idx_unsort = np.argsort(idx_sort)

        idx_sort = torch.from_numpy(idx_sort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_sort)
        sent = sent.index_select(1, idx_sort)

        # Handling padding in Recurrent Networks
        sent_packed = nn.utils.rnn.pack_padded_sequence(sent, sent_len_sorted)
        sent_output = self.enc_lstm(sent_packed)[0]  # seqlen x batch x 2*nhid
        sent_output = nn.utils.rnn.pad_packed_sequence(sent_output)[0]

        # Un-sort by length
        idx_unsort = torch.from_numpy(idx_unsort).cuda() if self.is_cuda() \
            else torch.from_numpy(idx_unsort)
        sent_output = sent_output.index_select(1, idx_unsort)

        # Pooling
        if self.pool_type == "mean":
            sent_len = torch.FloatTensor(sent_len.copy()).unsqueeze(1).cuda()
            emb = torch.sum(sent_output, 0).squeeze(0)
            emb = emb / sent_len.expand_as(emb)
        elif self.pool_type == "max":
            if not self.max_pad:
                sent_output[sent_output == 0] = -1e9
            emb = torch.max(sent_output, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2

        return emb

    def set_w2v_path(self, w2v_path):
        self.w2v_path = w2v_path

    def get_word_dict(self, sentences, tokenize=True):
        # create vocab of words
        word_dict = {}
        sentences = [s.split() if not tokenize else self.tokenize(s) for s in sentences]
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict[self.bos] = ''
        word_dict[self.eos] = ''
        return word_dict

    def get_w2v(self, word_dict):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with w2v vectors
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.fromstring(vec, sep=' ')
        print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
        return word_vec

    def get_w2v_k(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        # create word_vec with k first w2v vectors
        k = 0
        word_vec = {}
        with open(self.w2v_path, encoding='utf-8') as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if k <= K:
                    word_vec[word] = np.fromstring(vec, sep=' ')
                    k += 1
                if k > K:
                    if word in [self.bos, self.eos]:
                        word_vec[word] = np.fromstring(vec, sep=' ')

                if k > K and all([w in word_vec for w in [self.bos, self.eos]]):
                    break
        return word_vec

    def build_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        word_dict = self.get_word_dict(sentences, tokenize)
        self.word_vec = self.get_w2v(word_dict)
        print('Vocab size : %s' % (len(self.word_vec)))

    # build w2v vocab with k most frequent words
    def build_vocab_k_words(self, K):
        assert hasattr(self, 'w2v_path'), 'w2v path not set'
        self.word_vec = self.get_w2v_k(K)
        print('Vocab size : %s' % (K))

    def update_vocab(self, sentences, tokenize=True):
        assert hasattr(self, 'w2v_path'), 'warning : w2v path not set'
        assert hasattr(self, 'word_vec'), 'build_vocab before updating it'
        word_dict = self.get_word_dict(sentences, tokenize)

        # keep only new words
        for word in self.word_vec:
            if word in word_dict:
                del word_dict[word]

        # udpate vocabulary
        if word_dict:
            new_word_vec = self.get_w2v(word_dict)
            self.word_vec.update(new_word_vec)
        else:
            new_word_vec = []
        print('New vocab size : %s (added %s words)'% (len(self.word_vec), len(new_word_vec)))

    def get_batch(self, batch):
        # sent in batch in decreasing order of lengths
        # batch: (bsize, max_len, word_dim)
        embed = np.zeros((len(batch[0]), len(batch), self.word_emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[j, i, :] = self.word_vec[batch[i][j]]

        return torch.FloatTensor(embed)

    def tokenize(self, s):
        from nltk.tokenize import word_tokenize
        if self.moses_tok:
            s = ' '.join(word_tokenize(s))
            s = s.replace(" n't ", "n 't ")  # HACK to get ~MOSES tokenization
            return s.split()
        else:
            return word_tokenize(s)

    def prepare_samples(self, sentences, bsize, tokenize, verbose):
        sentences = [[self.bos] + s.split() + [self.eos] if not tokenize else
                     [self.bos] + self.tokenize(s) + [self.eos] for s in sentences]
        n_w = np.sum([len(x) for x in sentences])

        # filters words without w2v vectors
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in self.word_vec]
            if not s_f:
                import warnings
                warnings.warn('No words in "%s" (idx=%s) have w2v vectors. \
                               Replacing by "</s>"..' % (sentences[i], i))
                s_f = [self.eos]
            sentences[i] = s_f

        lengths = np.array([len(s) for s in sentences])
        n_wk = np.sum(lengths)
        if verbose:
            print('Nb words kept : %s/%s (%.1f%s)' % (
                        n_wk, n_w, 100.0 * n_wk / n_w, '%'))

        # sort by decreasing length
        lengths, idx_sort = np.sort(lengths)[::-1], np.argsort(-lengths)
        sentences = np.array(sentences)[idx_sort]

        return sentences, lengths, idx_sort

    def encode(self, sentences, bsize=64, tokenize=True, verbose=False):
        tic = time.time()
        sentences, lengths, idx_sort = self.prepare_samples(
                        sentences, bsize, tokenize, verbose)

        embeddings = []
        for stidx in range(0, len(sentences), bsize):
            batch = self.get_batch(sentences[stidx:stidx + bsize])
            if self.is_cuda():
                batch = batch.cuda()
            with torch.no_grad():
                batch = self.forward((batch, lengths[stidx:stidx + bsize])).data.cpu().numpy()
            embeddings.append(batch)
        embeddings = np.vstack(embeddings)

        # unsort
        idx_unsort = np.argsort(idx_sort)
        embeddings = embeddings[idx_unsort]

        if verbose:
            print('Speed : %.1f sentences/s (%s mode, bsize=%s)' % (
                    len(embeddings)/(time.time()-tic),
                    'gpu' if self.is_cuda() else 'cpu', bsize))
        return embeddings

    def visualize(self, sent, tokenize=True):

        sent = sent.split() if not tokenize else self.tokenize(sent)
        sent = [[self.bos] + [word for word in sent if word in self.word_vec] + [self.eos]]

        if ' '.join(sent[0]) == '%s %s' % (self.bos, self.eos):
            import warnings
            warnings.warn('No words in "%s" have w2v vectors. Replacing \
                           by "%s %s"..' % (sent, self.bos, self.eos))
        batch = self.get_batch(sent)

        if self.is_cuda():
            batch = batch.cuda()
        output = self.enc_lstm(batch)[0]
        output, idxs = torch.max(output, 0)
        # output, idxs = output.squeeze(), idxs.squeeze()
        idxs = idxs.data.cpu().numpy()
        argmaxs = [np.sum((idxs == k)) for k in range(len(sent[0]))]

        # visualize model
        import matplotlib.pyplot as plt
        x = range(len(sent[0]))
        y = [100.0 * n / np.sum(argmaxs) for n in argmaxs]
        plt.xticks(x, sent[0], rotation=45)
        plt.bar(x, y)
        plt.ylabel('%')
        plt.title('Visualisation of words importance')
        plt.show()

        return output, idxs


class Preprocess:
    def __init__(self):
        pass

    def remove_non_ascii(self, t):
        return ''.join([x if ord(x) < 127 else ' ' for x in t])

    def clean_table_array(self, tarr):
        return [[self.remove_non_ascii(x) for x in row] for row in tarr]


class CellDatasetInMemory(Dataset):
    def __init__(self, cells, senc, logger=None):
        #         self.file_list = self.zf.filelist
        self.cells = cells
        self.num_items = len(cells)
        self.senc = senc
        self.logger = logger

    def __getitem__(self, index):
        cell = self.cells[index]
        # print(cell)
        context = cell['context']
        value = cell['target']
        value_feat = np.array(cell['feat'])

        if self.senc:
            context_vecs = np.array([self.senc[x] for x in context])
            value_vecs = np.array([self.senc[value]])
        else:
            context_vecs = [0]
            value_vecs = [0]

        return context_vecs, value_vecs, value_feat

    def __len__(self):
        return self.num_items


class TableCellSample:
    def __init__(self, target_p, min_row, min_col, window, logger=None):
        self.prep = Preprocess()
        self.logger = logger
        self.target_p = target_p
        self.min_row = min_row
        self.min_col = min_col
        self.window = window

    def sample(self, table_array, feature_array, annotation_array):
        table_array = self.prep.clean_table_array(table_array)
        table_array = np.array(table_array)
        n = table_array.shape[0]
        if n < self.min_row:
            return []
        m = table_array.shape[1]
        if m < self.min_col:
            return []

        non_zero_inds = np.where(table_array != '')
        size = len(non_zero_inds[0])
        if size == 0:
            return []

        if self.target_p == 1.0:
            targets_i = non_zero_inds[0]
            targets_j = non_zero_inds[1]
        else:
            sample_size = int(size * self.target_p) + 1
            sample = np.random.choice(np.arange(size), sample_size, replace=False).tolist()
            targets_i = non_zero_inds[0][sample]
            targets_j = non_zero_inds[1][sample]

        sample_cells = []
        for i, j in zip(targets_i, targets_j):
            value = table_array[i, j]
            if feature_array is not None:
                feat = feature_array[i][j]
            else:
                feat = [0]
            if annotation_array is not None:
                ann = annotation_array[i][j]
            else:
                ann = None
            context = [None for _ in range(self.window * 4)]
            # print(len(context))
            ind = 0
            for wi in range(-self.window, self.window + 1):
                for wj in range(-self.window, self.window + 1):
                    if wi == 0 and wj == 0:
                        continue
                    if wi != 0 and wj != 0:
                        continue
                    ii = i + wi
                    jj = j + wj
                    # print(wi, wj, ind, ii,jj)
                    if 0 <= ii < n and 0 <= jj < m:
                        context[ind] = table_array[ii, jj]
                    ind += 1
            sample_cells.append(dict(target=value,
                                     context=context,
                                     feat=feat,
                                     ann=ann))
        return sample_cells


def __get_vec(line):
    args = line.split(' ', 1)
    w = args[0]
    v = np.array([float(x) for x in args[1].strip().split()])
    # assert len(v) == 300
    res = (w, v)
    return res


def load_WE(w2v_path, vocab_size):
    unk = '__UNK__'
    bs = '<s>'
    es = '</s>'
    lines = []
    with open(w2v_path, encoding='utf-8') as infile:
        for li, line in enumerate(infile):
            if li > vocab_size and line[:3] != bs and line[:4] != es:
                continue
            lines.append(line)
    print('loading word embeddings...')
    p = Pool(NUM_THREADS)
    res = p.map(__get_vec, lines)
    p.terminate()
    del lines
    print('creating dict...')
    res = dict(res)
    print('embeddings loaded!')
    return res, bs, es


class SentEnc():
    def __init__(self, model_path, w2v_path, vocab_size=100000, device='cpu', hp=False):
        self.vocab = set()
        self.device = device
        self.hp = hp

        self.tokenizer = word_tokenize
        self.vocab_size = vocab_size
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
        self.sent_model = InferSent(params_model).to(self.device).eval()
        self.sent_model.load_state_dict(torch.load(model_path, map_location=device))
        if hp:
            self.sent_model = self.sent_model.half()
        # self.sent_model = self.sent_model.cuda()
        # self.sent_model.set_w2v_path(w2v_path)
        self.init_vocab(w2v_path)
        self.cache = dict()

    def init_vocab(self, w2v_path):
        self.w2v, self.bs, self.es = load_WE(w2v_path, self.vocab_size)
        self.vocab.update(self.w2v.keys())
        # sentences = list(self.vocab) + [self.unk, self.bs, self.es]
        # self.sent_model.build_vocab(sentences, tokenize=True)

    def prune_sent(self, sent):
        if sent is None:
            sent_tok = []
        else:
            sent_tok = self.tokenizer(sent)
        sent_tok = [x for x in sent_tok if x in self.w2v]
        sent_rec = ' '.join(sent_tok)
        return sent_rec

    def get_number_encoding(self, num):
        d = 256
        device = 'cpu'
        # consider 4 decimal points
        a = int(num)
        b = int((num - a) * 1e2)

        a = str(a)[::-1]
        b = str(b)

        J = torch.arange(0, d)
        J_even = (J % 2 == 0).float().to(device)
        J = J.float().to(device)
        Ia = torch.arange(0, len(a)).float().to(device)
        Ib = (torch.arange(0, len(b)) + 1).float().to(device)
        A = torch.FloatTensor([float(x) for x in a]).to(device)
        B = torch.FloatTensor([float(x) for x in b]).to(device)

        J = J.float()

        A = torch.cat([A.view(1, -1)] * d, dim=0).T
        Ia = torch.cat([Ia.view(1, -1)] * d, dim=0).T

        B = torch.cat([B.view(1, -1)] * d, dim=0).T
        Ib = torch.cat([Ib.view(1, -1)] * d, dim=0).T

        resA = A * (2 ** Ia) / 10 * (J_even * torch.sin(Ia / (10000 ** (J / d))) + (1 - J_even) * torch.cos(Ia / (10000 ** ((J - 1) / d))))
        resB = B * (2.0 ** (-Ib)) / 10 * (J_even * torch.sin(-Ib / (10000 ** (J / d))) + (1 - J_even) * torch.cos(-Ib / (10000 ** ((J - 1) / d))))

        res = torch.sum(resA, axis=0) + torch.sum(resB, axis=0)
        res = res / (len(a) + len(b))
        res = res.numpy()
        if d < 4096:
            res = np.append(res, np.zeros(4096 - d)).astype('float32')
        return res

    def is_number(self, string):
        # try:
        #     num = abs(float(string))
        #     return num
        # except:
        #     return None
        string = string.strip()
        if re.match('^[-+]?\d+\.\d+$', string): return abs(float(string))
        if re.match('^[-+]?\d+$', string) and len(string) < 5: return abs(int(string))
        if re.match('^[-+]?[\d,]+$', string) and len(string) < 5: return abs(int(string.replace(',', '')))
        return None

    def get_text_encoding(self, sent):
        sent_tok = sent.split()
        s = np.array([self.w2v[self.bs]] + [self.w2v[x] for x in sent_tok] + [self.w2v[self.es]])
        l = len(s)
        s = s.reshape([l, 1, 300])
        s = torch.from_numpy(s).float()
        if self.hp:
            s = s.half()
        s = s.to(self.device)
        l = np.array([l], dtype='int64')

        # s = ' '.join(s)
        with torch.no_grad():
            res = self.sent_model.forward((s, l))
            # # v = torch.zeros([1,4096])
            if res.shape[1] != 4096:
                print(v, s)
            v = res.cpu().detach().numpy()[0]
            del res
            return v

    def cache_sentences(self, sentences):
        self.sent_cache = set()
        self.num_cache = set()
        for s in sentences:
            if s is None: s = ''
            if REG:
                num = self.is_number(s)
                if num is not None:
                    self.num_cache.add((num, s))
                else:
                    pruned_s = self.prune_sent(s)
                    self.sent_cache.add(pruned_s)
            else:
                pruned_s = self.prune_sent(s)
                self.sent_cache.add(pruned_s)
        print(f'initialize {len(self.sent_cache)} text sentences...')
        for s in tqdm.tqdm(self.sent_cache):
            self.cache[s] = self.get_text_encoding(s)
        print(f'initialize {len(self.num_cache)} numeric sentences...')
        self.num_cache = list(self.num_cache)
        for n, n_str in tqdm.tqdm(self.num_cache):
            self.cache[n_str] = self.get_number_encoding(n)
        # p = Pool(NUM_THREADS)
        # vecs = p.map(self.get_number_encoding, self.num_cache)
        # p.terminate()
        # for n, v in zip(self.num_cache, vecs):
        #     self.cache[str(num)] = v

    def __getitem__(self, sent):
        if sent is None: sent = ''
        if REG:
            num = self.is_number(sent)
            if num is not None:
                num_str = sent
                return self.cache[num_str]
            else:
                pruned_s = self.prune_sent(sent)
                return self.cache[pruned_s]
        pruned_s = self.prune_sent(sent)
        return self.cache[pruned_s]

        # return np.zeros((4096,), dtype='float32')


class SentEncWEAvg():
    def __init__(self, model_path, w2v_path, vocab_size=100000, device='cpu', hp=False):
        self.vocab = set()
        self.tokenizer = word_tokenize
        self.vocab_size = vocab_size
        self.init_vocab(w2v_path)

    def init_vocab(self, w2v_path):
        self.w2v = dict()
        with open(w2v_path, encoding='utf-8') as infile:
            for li, line in enumerate(infile):
                args = line.split(' ', 1)
                w = args[0]
                if li > self.vocab_size:
                    break
                v = np.array([float(x) for x in args[1].strip().split()])
                self.w2v[w] = v
                assert len(v) == 300
        self.vocab.update(self.w2v.keys())
        # sentences = list(self.vocab) + [self.unk, self.bs, self.es]
        # self.sent_model.build_vocab(sentences, tokenize=True)

    def __getitem__(self, sent):
        if sent is None:
            sent_tok = []
        else:
            sent_tok = self.tokenizer(sent)
        sent_tok = [x for x in sent_tok if x in self.w2v]

        if len(sent_tok) == 0:
            return np.zeros((300,))
        v = np.mean(np.array([self.w2v[x] for x in sent_tok]), axis=0)
        return v


def pack_seq(X, lens):
    idx_sort, lens_sorted = torch.sort(-lens)[::-1]  # , np.argsort(-lens)
    lens_sorted = -lens_sorted
    idx_unsort = torch.argsort(idx_sort)

    X = X.index_select(0, idx_sort)

    X_packed = torch.nn.utils.rnn.pack_padded_sequence(X, lens_sorted, batch_first=True)

    return X_packed, idx_unsort, idx_sort


def unpack_seq(X_packed, idx_unsort):
    X = torch.nn.utils.rnn.pad_packed_sequence(X_packed, batch_first=True)[0]
    return X.index_select(0, idx_unsort)


def ce_fit_iterative(ce_model, lr, loss, dataloader_train,
                     dataloader_dev, num_epochs, device, hp=False):
    adjust_p = lambda x: x.half() if hp else x
    loss_func = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(ce_model.parameters(), lr=lr)
    pbar = tqdm.tqdm(total=num_epochs * (len(dataloader_train) + len(dataloader_dev)))
    for e in range(num_epochs):
        bnum = 0
        eloss_train = 0
        for batch in dataloader_train:
            pbar.update(1)
            context_vecs, target_vecs, _ = batch
            context_vecs = adjust_p(context_vecs).to(device)
            target_vecs = adjust_p(target_vecs).to(device)
            # print(context_vecs.shape)
            # print(target_vecs.shape)
            bsize = context_vecs.shape[0]
            _, _, target_rec, context_rec = ce_model.forward(context_vecs, target_vecs)
            # print(target_rec)
            # print(context_rec.shape)
            loss = loss_func(target_rec, target_vecs.squeeze(1)) + loss_func(context_rec, torch.mean(context_vecs, dim=1))
            eloss_train += loss.item()
            ce_model.zero_grad()
            loss.backward()
            optimizer.step()
            bnum += 1
        eloss_train /= bnum

        bnum = 0
        eloss_dev = 0
        for batch in dataloader_dev:
            pbar.update(1)
            context_vecs, target_vecs, _ = batch
            context_vecs = context_vecs.to(device)
            target_vecs = target_vecs.to(device)
            bsize = context_vecs.shape[0]
            with torch.no_grad():
                _, _, target_rec, context_rec = ce_model.forward(context_vecs, target_vecs)
                loss = loss_func(target_rec, target_vecs.squeeze(1)) + loss_func(context_rec, torch.mean(context_vecs, dim=1))
            eloss_dev += loss.item()
            bnum += 1
        eloss_dev /= bnum

        yield ce_model, eloss_train, eloss_dev


def fe_fit_iterative(fe_model, lr, loss, dataloader_train,
                     dataloader_dev, num_epochs, device, hp=False):
    adjust_p = lambda x: x.half() if hp else x.float()
    loss_func = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(fe_model.parameters(), lr=lr)
    # pbar = tqdm.tqdm(total=num_epochs*(len(dataloader_train)+len(dataloader_dev)))
    pbar = tqdm.tqdm(total=num_epochs * len(dataloader_train))
    pbar.set_description(f'e:N/A,trl:N/A,devl:N/A')
    for e in range(num_epochs):
        bnum = 0
        eloss_train = 0
        for batch in dataloader_train:
            pbar.update(1)
            _, _, feats = batch
            feats = adjust_p(feats).to(device)
            bsize = feats.shape[0]
            feats = torch.log(feats + 1)
            feats_rec, _ = fe_model.forward(feats)
            loss = loss_func(feats_rec, feats)
            eloss_train += loss.item()
            fe_model.zero_grad()
            loss.backward()
            optimizer.step()
            bnum += 1
        eloss_train /= bnum

        # bnum = 0
        eloss_dev = 0
        # for batch in dataloader_dev:
        #     pbar.update(1)
        #     feats = adjust_p(feats).to(device)
        #     bsize = feats.shape[0]
        #     feats = torch.log(feats+1)

        #     with torch.no_grad():
        #         feats_rec, _ = fe_model.forward(feats)
        #         loss = loss_func(feats_rec, feats)
        #     eloss_dev += loss.item()
        #     bnum += 1
        # eloss_dev /= bnum
        pbar.set_description('e:{},trl:{:.2f},devl:{:.2f}'.format(e + 1, eloss_train, eloss_dev))
        yield fe_model, eloss_train, eloss_dev


###### CL functions

label2ind = ['attributes', 'data', 'header', 'metadata', 'derived', 'notes']


def split_train_test(tables, fold, dev_size):
    train_set = [(x['fname'], x['sname']) for x in fold['train']]
    train_set, dev_set = train_set[:-dev_size], train_set[-dev_size:]
    test_set = [(x['fname'], x['sname']) for x in fold['test']]

    train_inds = []
    dev_inds = []
    test_inds = []

    for ti, t in enumerate(tables):
        feat = t['feature_array']
        ann = t['annotations']
        fname = t['file_name'].lower()
        sname = t['table_id']
        tid = (fname, sname)
        if tid in train_set:
            train_inds.append(ti)
        elif tid in dev_set:
            dev_inds.append(ti)
        elif tid in test_set:
            test_inds.append(ti)

    tables_train = [tables[i] for i in train_inds]
    tables_dev = [tables[i] for i in dev_inds]
    tables_test = [tables[i] for i in test_inds]
    return tables_train, tables_dev, tables_test


def split_train_test_inds(tables, fold, dev_size):
    train_set = [(x['fname'], x['sname']) for x in fold['train']]
    train_set, dev_set = train_set[:-dev_size], train_set[-dev_size:]
    test_set = [(x['fname'], x['sname']) for x in fold['test']]

    train_inds = []
    dev_inds = []
    test_inds = []

    for ti, t in enumerate(tables):
        feat = t['feature_array']
        ann = t['annotations']
        fname = t['file_name'].lower()
        sname = t['table_id']
        tid = (fname, sname)
        if tid in train_set:
            train_inds.append(ti)
        elif tid in dev_set:
            dev_inds.append(ti)
        elif tid in test_set:
            test_inds.append(ti)
    return train_inds, dev_inds, test_inds


def get_nonempty_cells(tarr):
    n = tarr.shape[0]
    m = tarr.shape[1]
    non_zero_inds = np.where(tarr != '')
    size = len(non_zero_inds[0])
    if size == 0 or size > 50000:
        return 0, [], []

    targets_i = non_zero_inds[0]
    targets_j = non_zero_inds[1]
    return size, targets_i, targets_j


def get_annotations(ann_array, n, m):
    targets_i = []
    targets_j = []
    annotations = []
    if ann_array is None:
        return [], [], []
    for i in range(n):
        for j in range(m):
            ann = ann_array[i][j]
            if ann is not None:
                targets_i.append(i)
                targets_j.append(j)
                annotations.append(label2ind.index(ann))
    return annotations, targets_i, targets_j


def __get_contexts(ij, tarr, window):
    ind = 0
    i, j = ij
    t = tarr[i, j]
    n, m = tarr.shape
    contexts = ['' for _ in range(window * 4)]
    for wi in range(-window, window + 1):
        for wj in range(-window, window + 1):
            if wi == 0 and wj == 0:
                continue
            if wi != 0 and wj != 0:
                continue
            ii = i + wi
            jj = j + wj
            if 0 <= ii < n and 0 <= jj < m:
                c = tarr[ii, jj]
                contexts.append(c)
            ind += 1
    return contexts


def get_cevectarr2(tarr, ce_model, senc, device, window, senc_dim=4096):
    n, m = tarr.shape
    size, targets_i, targets_j = get_nonempty_cells(tarr)
    res = np.zeros([n, m, 2 * ce_model.encdim])
    targets = np.zeros([size, senc_dim])
    contexts = np.zeros([size, 4 * window, senc_dim])
    outer_ind = 0

    get_contexts = partial(__get_contexts, tarr=tarr, window=window)
    for i, j in zip(targets_i, targets_j):
        pass
    for ind, (i, j) in enumerate(zip(targets_i, targets_j)):
        t = tarr[i, j]
        targets[outer_ind, :] = senc[t]
        contexts[outer_ind, :, :]
    contexts = torch.from_numpy(contexts).float()
    targets = torch.from_numpy(targets).float()
    bsize = 1000
    nbatches = size // bsize
    if size % bsize > 0: nbatches += 1
    for i in range(nbatches):
        et_emb, ec_emb, _, _ = ce_model.forward(contexts[i * bsize:(i + 1) * bsize].to(device),
                                                targets[i * bsize:(i + 1) * bsize].to(device))
        res[(targets_i[i * bsize:(i + 1) * bsize], targets_j[i * bsize:(i + 1) * bsize])] = \
            torch.cat([et_emb, ec_emb], dim=-1).detach().cpu().numpy()
    return res


def get_cevectarr(tarr, ce_model, senc, device, window, senc_dim=4096):
    n, m = tarr.shape
    size, targets_i, targets_j = get_nonempty_cells(tarr)
    res = np.zeros([n, m, 2 * ce_model.encdim])
    if len(targets_i) == 0:
        return res
    targets = []
    contexts = []
    outer_ind = 0
    dummy = np.zeros(senc_dim)
    senc_cache = dict([(c, senc[c]) for c in tarr.flatten()])

    for i, j in zip(targets_i, targets_j):
        t = tarr[i, j]
        targets.append(senc_cache[t])
        ind = 0
        temp_ctx = [dummy for _ in range(window * 4)]
        for wi in range(-window, window + 1):
            for wj in range(-window, window + 1):
                if wi == 0 and wj == 0:
                    continue
                if wi != 0 and wj != 0:
                    continue
                ii = i + wi
                jj = j + wj
                if 0 <= ii < n and 0 <= jj < m:
                    c = tarr[ii, jj]
                    temp_ctx[ind] = senc_cache[c]
                ind += 1
        contexts.append(temp_ctx)
        outer_ind += 1
    contexts = torch.from_numpy(np.stack(contexts, axis=0)).float()
    targets = torch.from_numpy(np.stack(targets, axis=0)).float()
    bsize = 10000
    nbatches = size // bsize
    if size % bsize > 0: nbatches += 1
    for i in range(nbatches):
        et_emb, ec_emb, _, _ = ce_model.forward(contexts[i * bsize:(i + 1) * bsize].to(device),
                                                targets[i * bsize:(i + 1) * bsize].to(device))
        res[(targets_i[i * bsize:(i + 1) * bsize], targets_j[i * bsize:(i + 1) * bsize])] = \
            torch.cat([et_emb, ec_emb], dim=-1).detach().cpu().numpy()
    return res


def get_fevectarr(feature_array, n, m, fe_model, device):
    # res = np.array([n,m,fe_model.encdim])
    # ii, jj = []
    # ff = []
    # for i in range(n):
    #     for j in range(m):
    #         ii.append(i)
    #         jj.append(j)
    #         ff.append(feature_array[i][j])
    feature_array = feature_array[:n, :m, :]
    n, m, fdim = feature_array.shape
    ff = feature_array.reshape(n * m, fdim)
    ff = np.log(ff + 1)
    ff = torch.from_numpy(ff).float().to(device)
    _, fenc = fe_model.forward(ff)
    fenc = fenc.detach().cpu().numpy()
    return fenc.reshape(n, m, -1)


def get_class_weights(train_tables):
    annotations = []
    for t in train_tables:
        tarr = t['table_array']
        ann_array = t['annotations']
        n, m = len(tarr), len(tarr[0])
        temp_annotations, _, _ = get_annotations(ann_array, n, m)
        annotations += temp_annotations
    classes, counts = np.unique(annotations, return_counts=True)
    sorted_inds = np.argsort(classes)
    counts = counts[sorted_inds]
    assert len(counts) == 6
    return 1 - counts / np.sum(counts)


def get_df(report):
    lines = [x for x in report.split('\n') if x != '']
    col_header = None
    row_header = []

    data = []
    for i, l in enumerate(lines):
        args = re.split('\s\s+', l.strip())
        if i == 0:
            col_header = args
            continue
        data.append([float(x) for x in args[1:]])
        row_header.append(args[0])
    df = pd.DataFrame(data, columns=col_header, index=row_header)
    return df
