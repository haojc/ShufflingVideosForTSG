import numpy as np
import os
import json
import pandas as pd
from collections import OrderedDict
import logging
import string

def get_word_embedding(word_embedding_path,wordtoix_path,ixtoword_path,extracted_word_fts_init_path):
    print('loading word features ...')
    wordtoix = np.load(wordtoix_path, allow_pickle=True).tolist()
    ixtoword = np.load(ixtoword_path, allow_pickle=True).tolist()
    word_fts_dict = np.load(word_embedding_path,allow_pickle=True).tolist()
    print('load over. extracting')
    word_num = len(wordtoix)
    extract_word_fts = np.random.uniform(-3,3,[word_num,300])
    count = 0
    for index in range(word_num):
        if ixtoword[index] in word_fts_dict:
            extract_word_fts[index] = word_fts_dict[ ixtoword[index] ]
            count = count + 1
    print('total {:d} words embedding loaded of {:d} words'.format(count,word_num))
    """
    sentence 分词方式未与后面统一时：
        anet: total 10273 words embedding loaded of 11126 words
    
    sentence 分词方式统一后：
        charades train+test:
                            total 1258 words embedding loaded of 1294 words
        anet train:         total 10253 words embedding loaded of 10651 words
             train+test:    total 13020 words embedding loaded of 13745 words
        tacos   train+test: total 1806 words embedding loaded of 1858 words
                train+test+val: total 1961 words embedding loaded of 2027 words
    """

    if not os.path.exists(extracted_word_fts_init_path):
        np.save(extracted_word_fts_init_path,extract_word_fts)
        print("Save ", extracted_word_fts_init_path)

def preProBuildWordVocab(logging,sentence_iterator, word_count_threshold=5): # borrowed this function from NeuralTalk
    logging.info('preprocessing word counts and creating vocab based on word count threshold {:d}'.format(word_count_threshold))
    word_counts = {} # count the word number
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1  # if w is not in word_counts, will insert {w:0} into the dict

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words from {:d} to {:d}'.format(len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector

def glove_txt_to_npy(glove_txt, glove_npy):
    if not os.path.exists(glove_npy):
        glove_dict = {}
        n = 0
        with open(glove_txt) as ifs:
            for line in ifs:
                line = line.strip()
                if not line:
                    continue
                row = line.split()
                token = row[0]
                """
                128261 'at' 'name@domain'
                total 2196017
                """
                glove_dict[token] = [x for x in row[-300:]]
                # data = [float(x) for x in row[-300:]]
                # if len(data) != 300:
                #     raise RuntimeError("wrong number of dimensions", token)
                n = n + 1
        np.save(glove_npy, glove_dict)
        print('save end')
    else:
        print("{:s} existed".format(glove_npy))

def word_preprocess(logger, options, train_annotation_json, test_annotation_json, val_annotation_json=None):
    train_annotation_dict = json.load(open(train_annotation_json, 'r'))
    test_annotation_dict = json.load(open(test_annotation_json, 'r'))
    if val_annotation_json is not None:
        val_annotation_dict = json.load(open(val_annotation_json, 'r'))

    sentence_list = []

    for vid in train_annotation_dict:
        annotation = train_annotation_dict[vid]
        for sentence in annotation['sentences']:
            sentence_list.append(sentence.lower().strip())  #小写 + 删去头尾多余空格

    for vid in test_annotation_dict:
        annotation = test_annotation_dict[vid]
        for sentence in annotation['sentences']:
            sentence_list.append(sentence.lower().strip())  #小写 + 删去头尾多余空格

    if val_annotation_json is not None:
        for vid in val_annotation_dict:
            annotation = val_annotation_dict[vid]
            for sentence in annotation['sentences']:
                sentence_list.append(sentence.lower().strip())  # 小写 + 删去头尾多余空格

    sentences = sentence_list

    """     string.punctuation:   !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~      """
    for c in string.punctuation:
        if c == ',':
            sentences = list(map(lambda x: x.replace(c, ' '), sentences))
        else:
            sentences = list(map(lambda x: x.replace(c, ''), sentences))
    sentences = list(map(lambda x: ' '.join(x.replace('\n', '').split()), sentences))

    wordtoix, ixtoword, _ = preProBuildWordVocab(logger, sentences, word_count_threshold=1)

    if not os.path.exists(options['ixtoword_path']):
        np.save(options['ixtoword_path'], ixtoword)
        print("Save ", options['ixtoword_path'])
    if not os.path.exists(options['wordtoix_path']):
        np.save(options['wordtoix_path'], wordtoix)
        print("Save ", options['wordtoix_path'])
    if not os.path.exists(options['word_fts_path']):
        get_word_embedding(options['word_embedding_path'], options['wordtoix_path'], options['ixtoword_path'],
                           options['word_fts_path'])
    else:
        word_emb_init = np.array(np.load(options['word_fts_path']).tolist(), np.float32)
        print(1)
    print("Process over.")

def verify_word_embedding(annotation_json,params):
    annotation_dict = json.load(open(annotation_json, 'r'))

    sentence_list = []

    for vid in annotation_dict:
        annotation = annotation_dict[vid]
        for sentence in annotation['sentences']:
            sentence_list.append(sentence.lower().strip())  #小写 + 删去头尾多余空格
    sentences = sentence_list
    """
    string.punctuation:   !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    """
    for c in string.punctuation:
        if c == ',':
            sentences = list(map(lambda x: x.replace(c, ' '), sentences))
        else:
            sentences = list(map(lambda x: x.replace(c, ''), sentences))
    sentences = list(map(lambda x: ' '.join(x.replace('\n', '').split()), sentences))

    ixtoword = np.load(params['ixtoword_path'], allow_pickle=True).tolist()
    wordtoix = np.load(params['wordtoix_path'],allow_pickle=True).tolist()
    word_emb_init = np.array(np.load(params['word_fts_path']).tolist(), np.float32)

    MAX_SENTENCE_LEN = 25
    sentence_idxes = list(map(lambda x: [wordtoix[word] for word in x.lower().split(' ') if word in wordtoix], sentences))
    len_sentences = list(map(lambda x: len(x), sentence_idxes))
    print("Max sentence len: {:d}, Min len: {:d}, Mean: {:.2f}".format(max(len_sentences), min(len_sentences), sum(len_sentences)/len(len_sentences)))
    """
        charades train: Max sentence len: 10, Min len: 2, Mean: 6.21
        anet train:   Max sentence len: 73, Min len: 6, Mean: 14.22
    
    sentence分词方式统一后：
        Sentence Len    Max     Min     Mean
        anet train:     73      4      13.50
             test:      82      3      12.83
        tacos train:    81      1       8.63
                val:    202     1       9.03
                test:   141     1       8.92
        charades train: 11      2       6.21
                test:   10      2       6.24
    """

    pad_sentence_idxes = list(map(
        lambda x: np.pad(np.array(x),(0,MAX_SENTENCE_LEN-len(x))).tolist() if len(x)<MAX_SENTENCE_LEN else np.array(x)[:MAX_SENTENCE_LEN],
        sentence_idxes))

    pad_sentence_idx = pad_sentence_idxes[0]
    sentence_features = list(map(lambda x: word_emb_init[x], pad_sentence_idx))

    print(1)

if __name__ == '__main__':
    glove_txt = 'data/glove.840B.300d.txt'
    glove_npy = 'data/glove.840B.300d_dict.npy'
    glove_txt_to_npy(glove_txt, glove_npy)

    dataset = 'charades'

    options = OrderedDict()
    if dataset== 'charades':
        options['word_embedding_path'] = 'data/glove.840B.300d_dict.npy'
        options['wordtoix_path'] = 'grounding/Charades/words/wordtoix.npy'
        options['ixtoword_path'] = 'grounding/Charades/words/ixtoword.npy'
        options['word_fts_path'] = 'grounding/Charades/words/word_glove_fts_init.npy'
        train_json_file = 'data/Charades/train.json'
        test_json_file = 'data/Charades/test.json'
        val_json_file = None
    elif dataset == 'anet':
        options['word_embedding_path'] = 'data/glove.840B.300d_dict.npy'
        options['wordtoix_path'] = 'grounding/ActivityNet/words/wordtoix.npy'
        options['ixtoword_path'] = 'grounding/ActivityNet/words/ixtoword.npy'
        options['word_fts_path'] = 'grounding/ActivityNet/words/word_glove_fts_init.npy'
        train_json_file = 'data/ActivityNet/train.json'
        test_json_file = 'data/ActivityNet/val_merge.json'
        val_json_file = None
    elif dataset == 'tacos':
        options['word_embedding_path'] = 'data/glove.840B.300d_dict.npy'
        options['wordtoix_path'] = 'grounding/TACoS/words/wordtoix.npy'
        options['ixtoword_path'] = 'grounding/TACoS/words/ixtoword.npy'
        options['word_fts_path'] = 'grounding/TACoS/words/word_glove_fts_init.npy'
        train_json_file = 'data/TACOS/train_f.json'
        test_json_file = 'data/TACOS/test_f.json'
        val_json_file = 'data/TACOS/val_f.json'

    word_preprocess(logging,options, train_json_file,test_json_file, val_json_file)
    """
    5.26
    json文件中word embedding转换，但是glove中缺失部分word. 
    total 1090 words embedding loaded of 1138 words
    缺失部分的word embedding随机生成
    np.random.uniform(-3,3,[word_num,300])
    未来考虑自己训练word embedding
    
    8.4
    注意这里计数word时，对sentence的分词操作和其他地方保持一致。
    尤其anet的sentence中含有 \n, =, !, [等
    """

    verify_word_embedding(train_json_file,options)
    verify_word_embedding(test_json_file, options)