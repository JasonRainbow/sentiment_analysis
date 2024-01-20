from ThreeSentiment.build_data import convert_text_to_token
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn import metrics
import os
import gc
from sql_dao.sql_utils import get_conn, query_polarity_sql, delete_polarity_sql, insert_polarity
from pymysql.err import ProgrammingError, MySQLError


# 复用模型结构
class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(parent_dir + '/chinese_wwm_ext_pytorch')  # /roberta-wwm-ext pretrain/
        for param in self.bert.parameters():
            param.requires_grad = True  # 所有参数求梯度
        self.fc = nn.Linear(768, num_classes)  # 768 -> 6

    def forward(self, x, token_type_ids, attention_mask):
        context = x  # 输入的句子
        types = token_type_ids
        mask = attention_mask  # 对padding部分进行mask，和句子相同size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, token_type_ids=types, attention_mask=mask)
        out = self.fc(pooled)  # 得到2分类概率
        return out


LABEL_DICT = {0: '消极', 1: '中立', 2: '积极'}  # 标签映射表
SEQ_LENGTH = 128

parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # 父目录


def load_model():
    PATH = parent_dir + '/model/roberta_three_sentiment_model.pth'
    TOKENIZER = BertTokenizer.from_pretrained(parent_dir + "/chinese_wwm_ext_pytorch")  # 模型[roberta-wwm-ext]所在的目录名称
    # 加载模型
    MODEL = Model(num_classes=3)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = MODEL.to(DEVICE)
    MODEL.load_state_dict(torch.load(PATH))
    # print('模型加载完毕')
    return TOKENIZER, MODEL, DEVICE


#  对单个句子进行情感分析 返回情感极性结果
def predictSingle(sentence, TOKENIZER, MODEL, DEVICE):
    ids = []
    types = []
    masks = []
    cur_ids, cur_type, cur_mask = convert_text_to_token(TOKENIZER, sentence, seq_length=SEQ_LENGTH)
    # print(cur_ids, cur_type, cur_mask)
    ids.append(cur_ids)
    types.append(cur_type)
    masks.append(cur_mask)
    cur_ids, cur_type, cur_mask = torch.LongTensor(np.array(ids)).to(DEVICE), torch.LongTensor(np.array(types)).to(
        DEVICE), torch.LongTensor(np.array(masks)).to(DEVICE)  # 数据构造成tensor形式
    # print(cur_ids, cur_type, cur_mask)
    with torch.no_grad():
        y_ = MODEL(cur_ids, token_type_ids=cur_type, attention_mask=cur_mask)
        pred = y_.max(-1, keepdim=True)[1]  # 取最大值
        cur_pre = LABEL_DICT[int(pred[0][0].cuda().data.cpu().numpy())]  # 预测的情绪
    return cur_pre


#  利用sklearn计算模型的性能指标accuracy(准确率）、precision(精确率）、recall(召回率)、f1score(F1值)、auc
def model_all_target_test():
    TOKENIZER, MODEL, DEVICE = load_model()
    f = open('../data/three_sentiment_test.csv', 'r', encoding='utf-8')
    data = pd.read_csv(f, sep="\t")
    y_true = []
    y_pred = []
    y_score = []  # 概率值
    record_num = data.shape[0]  # 返回行数
    #  遍历所有行
    for i in range(record_num):
        record = data.iloc[i, :]
        y_true.append(record['label'])
        ids = []
        types = []
        masks = []
        cur_ids, cur_type, cur_mask = convert_text_to_token(TOKENIZER, record['review'], seq_length=SEQ_LENGTH)
        ids.append(cur_ids)
        types.append(cur_type)
        masks.append(cur_mask)
        cur_ids, cur_type, cur_mask = torch.LongTensor(np.array(ids)).to(DEVICE), torch.LongTensor(np.array(types)).to(
            DEVICE), torch.LongTensor(np.array(masks)).to(DEVICE)  # 数据构造成tensor形式

        with torch.no_grad():
            y_ = MODEL(cur_ids, token_type_ids=cur_type, attention_mask=cur_mask)
            # probability = MODEL.forward(record['review'], token_type_ids=cur_type, attention_mask=cur_mask)  #计算得到概率值
            pred = y_.max(-1, keepdim=True)[1]  # 取最大值
            cur_pre = int(pred[0][0].cuda().data.cpu().numpy())  # 预测的情绪
        y_pred.append(cur_pre)
        # y_score.append(probability)

    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='micro')
    recall = metrics.recall_score(y_true, y_pred, average='micro')
    f1score = metrics.f1_score(y_true, y_pred, average='micro')
    # fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    # auc = metrics.auc(fpr, tpr)
    # return accuracy, precision, recall, f1score, auc
    return accuracy, precision, recall, f1score


def polarity_analyze_by_workId(workId):
    print("workId：%d ，开始进行极性情感分析" % workId)
    TOKENIZER, MODEL, DEVICE = load_model()
    conn = get_conn()
    sql1 = "select distinct country, platform, postTime from raw_comment where workId = %d" % workId
    # 分别获取workId号作品的评论所属的国家列表、平台列表、发布时间列表
    df = pd.read_sql(sql1, con=conn)
    for i in range(len(df)):
        polarity_analyze(workId, df["country"][i], df["platform"][i],
                          df["postTime"][i], TOKENIZER, MODEL, DEVICE, conn)
    conn.close()
    del TOKENIZER, MODEL, DEVICE, conn
    gc.collect()
    print("workId：%d ，完成极性情感分析" % workId)
    return True


def polarity_analyze(workId, country, platform, post_time, TOKENIZER, MODEL, DEVICE, conn):
    sql = """
            select translated from raw_comment 
            where workId = {} and country = "{}" and platform = "{}"
            and postTime = "{}";
        """.format(workId, country, platform, post_time)
    comments = pd.read_sql(sql=sql, con=conn)
    if len(comments) == 0:
        # print("没有评论")
        return
    comments = comments["translated"].tolist()  # 获取查询到的评论列表
    positive = 0
    negative = 0
    neutrality = 0
    for comment in comments:
        if comment.strip() == 0:
            continue
        res = predictSingle(comment, TOKENIZER, MODEL, DEVICE)
        if res == "积极":
            positive += 1
        elif res == "消极":
            negative += 1
        elif res == "中立":
            neutrality += 1

    cursor = conn.cursor()
    cursor.execute(query_polarity_sql.format(workId, country, platform, post_time))
    if len(cursor.fetchall()) > 0:
        cursor.execute(delete_polarity_sql.format(workId, country, platform, post_time))
    try:
        # print("写入数据库")
        insert_polarity(workId, country, platform, post_time, positive, negative, neutrality, conn)
    except (MySQLError, ProgrammingError):
        print("插入失败，有错误")
    finally:
        conn.commit()  # 提交修改


def my_test():
    TOKENIZER, MODEL, DEVICE = load_model()
    polarity = predictSingle('我觉得这一般啊，', TOKENIZER, MODEL, DEVICE)
    print(polarity)


if __name__ == '__main__':
    my_test()
    # model_all_target_test()
    # accuracy, precision, recall, f1score, auc = model_all_target_test()
    # accuracy, precision, recall, f1score = model_all_target_test()
    # print(accuracy, precision, recall, f1score)
    pass
