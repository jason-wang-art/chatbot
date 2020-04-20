import jieba
import os
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np


# from sklearn.model_selection import train_test_split


class PreProcess(object):
    def __init__(self, file_path, max_length=25, samples_num=10000, is_char=False, random_choose=False,
                 filter_max_lenght=False):
        """

        :param file_path:
        :param max_length 语句的最大长度
        :param samples_num:
        :param is_char:
        :param random_choose: 随机挑选样本
        """
        self.is_char = is_char
        # 将分词后的语句写入新文件
        abs_path = os.path.abspath(file_path)
        save_path = abs_path.split('.tsv')[0] + '_jieba.tsv'
        if not os.path.exists(save_path):
            self._parse_file(file_path, save_path)
        q, a = self._load_parse_file(save_path)
        # 过滤掉长度过长的句子
        if filter_max_lenght:
            q, a, filter_length = self._filter_max_lenght(q, a, max_length)
            if samples_num > filter_length:
                samples_num = filter_length
        # load data
        self.q_tensor, self.q_tokenizer, self.a_tensor, self.a_tokenizer = self._load_data(q, a)
        # 过滤掉超过max_length 的语句
        # 取samples_num个用来训练
        self._choose_train_samples(random_choose, samples_num)

        # self._random_shuffle()
        # self.q_tensor_train, self.q_tensor_val, self.a_tensor_train, self.a_tensor_val = \
        #     train_test_split(self.q_tensor, self.a_tensor, test_size=0.2)

    def _filter_max_lenght(self, q, a, max_length):
        q_filter = []
        a_filter = []
        for q, a in zip(*[q, a]):
            if len(q) > max_length or len(a) > max_length:
                continue
            q_filter.append(q)
            a_filter.append(a)
        return q_filter, a_filter, len(q_filter)

    def _choose_train_samples(self, random_choose, samples_num):
        if random_choose:
            ids = np.random.choice(np.arange(self.length), samples_num, replace=False)
        else:
            ids = np.arange(samples_num)
        self.q_tensor = self.q_tensor[ids]
        self.a_tensor = self.a_tensor[ids]

    def _random_shuffle(self):
        ids = np.arange(self.length)
        np.random.shuffle(ids)
        self.q_tensor = self.q_tensor[ids]
        self.a_tensor = self.a_tensor[ids]

    def _parse_file(self, file_path, save_path):
        with open(file_path, 'r') as f:
            lines = self._purify(f.readlines())
            data = [line.split('\t') for line in lines]
            if self.is_char:
                pass
            else:
                data = [(jieba.lcut(qa[0]), jieba.lcut(qa[1])) for qa in data]
        f.close()
        with open(save_path, 'w') as f:
            for line in data:
                q, a = line
                content = ' '.join(q) + '\t' + ' '.join(a) + '\n'
                f.write(content)
        f.close()

    def _load_parse_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            data = [('<eos> ' + line.split('\t')[0] + ' <sos>', '<eos> ' + line.split('\t')[1].strip('\n') + ' <sos>')
                    for line in lines]
        f.close()
        return zip(*data)

    def tokenize(self, qa):
        tokenizer = Tokenizer(filters='', split=' ', num_words=None)
        tokenizer.fit_on_texts(qa)
        tensor = tokenizer.texts_to_sequences(qa)
        tensor = pad_sequences(tensor, padding='post')
        return tensor, tokenizer

    def val_process(self, sentence):
        words = jieba.cut(sentence)
        words = '<eos> ' + ' '.join(words) + ' <sos>'
        print(words)
        tensor = self.q_tokenizer.texts_to_sequences(words)
        tensor = pad_sequences(tensor, padding='post')
        return tensor.reshape((1, -1))

    def _purify(self, lines):
        return [self.filter_sentence(line) for line in lines]

    # 去掉一些停用词
    def filter_sentence(self, sentence):
        return sentence.replace('\n', '').replace(' ', '').replace('，', ',').replace('。', '.'). \
            replace('；', '：', ':').replace('？', '?').replace('！', '!').replace('“', '"'). \
            replace('”', '"').replace("‘", "'").replace("’", "'").replace('（', '(').replace('）', ')')

    def _load_data(self, q, a):
        q_tensor, q_tokenizer = self.tokenize(q)
        a_tensor, a_tokenizer = self.tokenize(a)
        return q_tensor, q_tokenizer, a_tensor, a_tokenizer

    @property
    def length(self):
        if len(self.q_tensor) != len(self.a_tensor):
            raise ValueError('data load error please check it!!!')
        return len(self.q_tensor)

    @property
    def q_vocab_size(self):
        return len(self.q_tokenizer.word_index) + 1

    @property
    def a_vocab_size(self):
        return len(self.a_tokenizer.word_index) + 1

    def test(self, lang, tensor):
        for t in tensor:
            if t != 0:
                print("%d ----> %s" % (t, lang.index_word[t]))

    @property
    def q_lenght(self):
        return len(self.q_tensor[-1])

    @property
    def a_lenght(self):
        return len(self.a_tensor[-1])

    def generate_data(self, bath_size):
        if bath_size <= 0:
            raise ValueError('batch_size should > 0')
        i = 0
        while True:
            # 简单实现
            q_tensor = []
            a_tensor = []
            for _ in range(bath_size):
                if i == 0:
                    self._random_shuffle()
                q = self.q_tensor[i]
                a = self.a_tensor[i]
                q_tensor.append(q)
                a_tensor.append(a)
                i = (i + 1) % self.length
            yield np.array(q_tensor), np.array(a_tensor)

process = PreProcess('./data/qingyun.tsv')
print(len(process.q_tensor))
data = process.val_process('你猜猜看我是谁')
print(data, data.shape)
#
# print(process.test(process.a_tokenizer, process.a_tensor[-1]))
# print(process.a_tensor[-1])

# generate = process.generate_data(10)
# q, a = next(generate)
# print(q.shape, a.shape)
