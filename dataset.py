import pandas as pd
import unicodedata
import re
from config import data_dir, MAX_LENGTH
import torchtext
import torch

#将unicode字符串标准化：
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# 规范化字符串
def normalizeString(s):
    s = s.lower().strip()
    s = unicodeToAscii(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r'[\s]+', " ", s)
    return s

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[0].startswith(eng_prefixes)

def filterPairs(pairs):
    return [[pair[1], pair[0]] for pair in pairs if filterPair(pair)]

data_df = pd.read_csv(data_dir + 'eng-fra.txt', encoding='UTF-8', sep='\t', header=None, names=['eng', 'fra'], index_col=False)

pairs = [[normalizeString(s) for s in line] for line in data_df.values]
pairs = filterPairs(pairs)

def get_dataset(pairs, src, targ):
    fields = [('src', src), ('targ', targ)]
    examples = []
    for fra, eng in pairs:
        examples.append(torchtext.legacy.data.Example.fromlist([fra, eng], fields))
    return examples, fields

# 将数据管道组织成与torch.utils.data.DataLoader相似的inputs, targets的输出形式
class DataLoader:
    def __init__(self, data_iter):
        self.data_iter = data_iter
        self.length = len(data_iter)

    def __len__(self):
        return self.length

    def __iter__(self):
        for batch in self.data_iter:
            yield (torch.transpose(batch.src, 0, 1), torch.transpose(batch.targ, 0, 1))