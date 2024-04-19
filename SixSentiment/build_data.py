import json
from transformers import BertTokenizer
from torch.utils.data import *
import torch
import numpy as np

SEQ_LENGTH = 200
BATCH_SIZE = 16
LABEL_DICT = {'fear': 0, 'neutral': 1, 'sad': 2, 'surprise': 3, 'angry': 4, 'happy': 5}  # 标签映射表


# 数据进行token化处理, seq_length表示接受的句子最大长度
def convert_text_to_token(tokenizer, sentence, seq_length):
    tokens = tokenizer.tokenize(sentence)  # 句子转换成token
    tokens = ["[CLS]"] + tokens + ["[SEP]"]  # token前后分别加上[CLS]和[SEP]
    # 生成 input_id, seg_id, att_mask
    ids1 = tokenizer.convert_tokens_to_ids(tokens)
    types = [0] * len(ids1)
    masks = [1] * len(ids1)
    # 句子长度统一化处理：截断或补全至seq_length
    if len(ids1) < seq_length:  # 补全
        ids = ids1 + [0] * (seq_length - len(ids1))  # [0]是因为词表中PAD的索引是0
        types = types + [1] * (seq_length - len(ids1))  # [1]表明该部分为PAD
        masks = masks + [0] * (seq_length - len(ids1))  # PAD部分，attention mask置为[0]
    else:  # 截断
        ids = ids1[:seq_length]
        types = types[:seq_length]
        masks = masks[:seq_length]
    assert len(ids) == len(types) == len(masks)
    return ids, types, masks


# 构造训练集和测试集的DataLoader
def genDataLoader(is_train):
    # 模型[roberta-wwm-ext]所在的目录名称
    TOKENIZER = BertTokenizer.from_pretrained("../chinese_wwm_ext_pytorch")
    TRAIN_DATA_PATH = '../data/usual_train.txt'
    TEST_DATA_PATH = '../data/usual_test_labeled.txt'
    if is_train:  # 构造训练集
        path = TRAIN_DATA_PATH
    else:  # 构造测试集
        path = TEST_DATA_PATH
    with open(path, encoding='utf8') as f:
        data = json.load(f)
    ids_pool = []
    types_pool = []
    masks_pool = []
    target_pool = []
    count = 0
    # 遍历构造每条数据
    for each in data:
        cur_ids, cur_type, cur_mask = convert_text_to_token(TOKENIZER, each['content'], seq_length=SEQ_LENGTH)
        ids_pool.append(cur_ids)
        types_pool.append(cur_type)
        masks_pool.append(cur_mask)
        cur_target = LABEL_DICT[each['label']]
        target_pool.append(cur_target)
        count += 1
        if count % 1000 == 0:
            print('已处理{}条'.format(count))
            # break
    # 构造loader
    data_gen = TensorDataset(torch.LongTensor(np.array(ids_pool)),
                             torch.LongTensor(np.array(types_pool)),
                             torch.LongTensor(np.array(masks_pool)),
                             torch.LongTensor(np.array(target_pool)))
    # print('shit')
    sampler = RandomSampler(data_gen)  # 全部采样并打乱顺序，返回下标
    loader = DataLoader(data_gen, sampler=sampler, batch_size=BATCH_SIZE)
    return loader
