from __future__ import absolute_import, division, print_function
from lib2to3.pgen2 import token

import logging
from io import open
from typing_extensions import ParamSpec
from regex import P
from transformers import AutoTokenizer,BartTokenizer
import pickle
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
 
class InputExample():#代表了数据的一组特征，InputExample对象会被用作模型的输入
    """A single set of features of data."""
    def __init__(self,token_a,token_b,label,domain,parent=None,child=None):
        self.token_a = token_a#token_a代表comment
        self.token_b = token_b#token_b代表replay
        self.label = label#数据的标签
        self.domain = domain#表示数据所属的领域
        self.parent = parent#表示数据的父节点
        self.child = child#表示数据的子节点

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label,domain=None,p=None,c=None):
        self.input_ids = input_ids#将文本转化为模型可理解的数字表示的序列，这些数字是词汇表中的文本的位置
        self.input_mask = input_mask#用于指示哪些位置是真实的输入，哪些是填充
        self.segment_ids = segment_ids#用于区分句子的顺序，标识哪些部分属于第一个句子，哪些部分属于第二个句子,此处区分评论和回复
        self.label = label
        self.domain = domain
        self.p = p
        self.c = c

class DataLoader__(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dirs):#返回训练集的一组 InputExample 对象,data_dirs 参数指定训练集数据的目录或路径
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dirs):#验证集
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dirs):#测试集
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):#获取数据集的标签列表
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
    
#StanceDataloader 是一个继承自 DataLoader__ 的子类，用于加载 Stance 数据集的训练、验证和测试集
class StanceDataloader(DataLoader__):
    def get_train_examples(self, data_dirs,topic):
        if str(topic)=='all':
            return self._create_examples(data_dirs=data_dirs, set_type='train',genre=topic)
        else: return self._create_examples_l(data_dirs=data_dirs, set_type='train',genre=topic)
#根据 topic 的取值是否为 'all'，选择调用 _create_examples 或 _create_examples_l 方法来创建相应的 InputExample
    def get_dev_examples(self, data_dirs,topic):
        if str(topic)=='all':
            return self._create_examples(data_dirs=data_dirs, set_type='dev',genre=topic)
        else: return self._create_examples_l(data_dirs=data_dirs, set_type='dev',genre=topic)

    def get_test_examples(self, data_dirs,topic):
        if str(topic)=='all':
            return self._create_examples(data_dirs=data_dirs, set_type='test',genre=topic)
        else: return self._create_examples_l(data_dirs=data_dirs, set_type='test',genre=topic)

    def get_data_examples(self, data_dirs,topic):#获取包含数据集所有样本的 InputExample 对象
        return self._create_examples_l(data_dirs=data_dirs, set_type='all',genre=topic)

    def _create_examples(self, data_dirs, set_type, genre='all'):#in-domain experiment
        examples = []

        data = pd.read_csv(data_dirs, index_col = False,encoding='latin-1')
        data = data.sort_values(by=['datetime'])
        if genre == 'all':#包括所有的主题
            data = data
            if set_type == 'train':
                data = data.iloc[0:int(len(data)*0.8)]
            elif set_type =='dev':
                data = data.iloc[int(len(data)*0.8):int(len(data)*0.9)]
            elif set_type == 'test':
                data = data.iloc[int(len(data)*0.9):int(len(data))]
            elif set_type == 'all':
                data = data
        
        texts_a = data['body_parent']
        texts_b = data['body_child']
        labels = data['label']
        domain = data['subreddit']
        p = data['author_parent']
        c = data['author_child']
        for t in enumerate(zip(texts_a,texts_b,labels,domain,p,c)):
            t = t[1]
            examples.append(InputExample(t[0],t[1],t[2],t[3],t[4],t[5]))

        return examples

    def _create_examples_l(self, data_dirs, set_type, genre=[]):#cross-domain experiment 
        examples = []

        data = pd.read_csv(data_dirs, index_col = False,encoding='latin-1')
        new_date = []
        data = data.sort_values(by=['datetime'])

        for gen in genre:#划分为子主题
            data2 = data[data['subreddit']==gen]
            if set_type == 'train':
                data2 = data2.iloc[0:int(len(data2)*0.8)]
            elif set_type =='dev':
                data2 = data2.iloc[int(len(data2)*0.8):int(len(data2)*0.9)]
            elif set_type == 'test':
                data2 = data2.iloc[int(len(data2)*0.9):int(len(data2))]
        
            texts_a = data2['body_parent']
            texts_b = data2['body_child']
            labels = data2['label']
            domain = data2['subreddit']
            p = data2['author_parent']
            c = data2['author_child']

            for t in enumerate(zip(texts_a,texts_b,labels,domain,p,c)):
                t = t[1]
                examples.append(InputExample(t[0],t[1],t[2],t[3],t[4],t[5]))

        return examples

#将输入的序列对 tokens_a 和 tokens_b 截断到指定的最大长度 max_length
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        tokens_a.pop()#否则，从 tokens_a 的末尾删除一个标记
'''def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) > 0:
            tokens_a.pop()  # 如果 tokens_a 的长度大于等于 tokens_b 的长度，从 tokens_a 中删除一个标记
        elif len(tokens_b) > 0:
            tokens_b.pop()  # 否则，从 tokens_b 中删除一个标记
        if len(tokens_a) == 0 and len(tokens_b) == 0:
            raise ValueError("Both tokens_a and tokens_b are empty, but total length exceeds max length.")'''

def convert_examples_to_features(examples, max_seq_length,
                                 tokenizer,unique_nodes_mapping=None,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    unique_nodes_mapping = pickle.load(open('utils/unique_nodes_n_mapping_bert.pkl', 'rb'))#加载唯一节点映射表
    features = []#初始化特征列表 features
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.token_a)

        tokens_b = None
        if example.token_b:#如果存在token_b，则对 token_b 进行标记化，结果存储在 tokens_b 中
            tokens_b = tokenizer.tokenize(example.token_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:#如果不存在 token_b，则将 tokens_a 截断到最大长度 max_seq_length - 2 的限制。
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
        
        # if not (len(tokens_a)+len(tokens_b)<200) and  (len(tokens_a)+len(tokens_b)>100):
        #     continue
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]#将 tokens_a 添加 [SEP] 标记，并将结果存储在 tokens 中。
        segment_ids = [sequence_a_segment_id] * len(tokens)
        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)
        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #if len(input_ids)>200:#
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)#指示输入序列中哪些标记是真实的有效标记，哪些是填充标记

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        features.append(
                InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label=example.label,
                            domain = example.domain,
                            p = unique_nodes_mapping[example.parent],
                            c = unique_nodes_mapping[example.child]
                            ))

    return features

def convert_examples_to_features_(examples, max_seq_length,
                                 tokenizer):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    unique_nodes_mapping = pickle.load(open('utils/unique_nodes_n_mapping_bert.pkl', 'rb'))

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        text = example.token_a + ' ' + str(tokenizer.sep_token) + ' ' + example.token_b

        inputs = tokenizer(text, padding='max_length', max_length=max_seq_len,truncation=True)

        assert len(inputs['input_ids']) == max_seq_len
        assert len(inputs['attention_mask']) == max_seq_len
 
        # print(example)
        features.append(
                InputFeatures(input_ids=inputs['input_ids'],
                              input_mask=inputs['attention_mask'],
                              segment_ids=None,
                              label=example.label,
                              domain = example.domain,
                              p=unique_nodes_mapping[example.parent],
                              c = unique_nodes_mapping[example.child]
                              ))
    return features

class SentGloveFeatures(object):
    def __init__(self,tokens,embeddings,input_mask,label,domain = None):
        self.text_a = tokens
        self.input_mask = input_mask#句子的输入掩码，指示哪些位置是真实的标记
        self.label = label
        self.domain = domain
        self.embeddings = embeddings

def get_all_words(examples,GLOVE_DIR):
    all_words=set()#初始化一个空集合 all_words，用于存储所有的单词
    embeddings_index = {}
    for (ex_index, example) in enumerate(examples):
        tokens_a = word_tokenize(example.token_a)#用 word_tokenize 函数对句子A和句子B进行分词，得到它们的单词列表
        tokens_b = word_tokenize(example.token_b)
        for a in tokens_a:#将句子A和句子B的所有单词添加到 all_words 集合中
            all_words.add(a)
        for b in tokens_b:
            all_words.add(b)

    with open('all_words.txt','w',encoding='utf8') as fs:
        for a in all_words:#遍历 all_words 集合中的每个单词 a，将其写入文件中，并在每个单词之间添加一个空格
            fs.write(a+" ")
        fs.close()
    
    #使用 "all_vectors.txt" 文件中的词向量
    with open('glove.6B.300d.txt',encoding='utf8') as f:
        with open('all_vectors.txt','wb') as fs:
            for line in f:
                values = line.split()
                word = values[0]
                if word in all_words:
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                    # fs.write(word+' '+str(coefs)+'\n')
                if word == 'unk':
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index['unk'] = coefs
                    # fs.write('unk'+' '+str(coefs)+'\n')
            pickle.dump(embeddings_index,fs)
            print('Finished')

def retrive_word_embedding(examples,  max_seq_length,
                           pad_on_left=False, pad_token='unk',
                            mask_padding_with_zero=True):
    features = []
    all_words_vectors={}
    with open('all_vectors.txt','rb') as f:
        all_words_vectors = pickle.load(f)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        tokens_a = word_tokenize(example.token_a)
        if len(tokens_a) > max_seq_length:
                tokens_a = tokens_a[:(max_seq_length)]
        input_mask = [1 if mask_padding_with_zero else 0] * len(tokens_a)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(tokens_a)
        words = tokens_a + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        seq_vectors = [(all_words_vectors[a] if a in all_words_vectors.keys() else all_words_vectors['unk']) for a in words]
        seq_vectors = np.array(seq_vectors)
        features.append(SentGloveFeatures(tokens_a,seq_vectors,input_mask,label=example.label, domain=example.domain))
    return features
#根据给定的领域和实验类型从数据集中获取相应的父节点和子节点作者信息，用于构建立场数据集。
def get_stance_dataset(domain = 'labeled_data',exp_type='train'):
    data = pd.read_csv(domain+".csv", index_col = False)
    data = data.sort_values(by=['datetime'])
    nodes = []
    if exp_type == 'train':
        data = data.iloc[0:int(len(data)*0.8)]
    elif exp_type =='dev':
        data = data.iloc[int(len(data)*0.8):int(len(data)*0.9)]
    elif exp_type == 'test':
        data = data.iloc[int(len(data)*0.9):int(len(data))]
    parent = (data['author_parent'])
    child = (data['author_child'])
    for p in enumerate(zip(parent,child)):
        p=p[1]
        nodes.append((p[0],p[1]))
    return nodes

