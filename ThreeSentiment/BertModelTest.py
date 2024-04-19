import time

from ThreeSentiment.build_data import genDataLoader, genDataLoaderFromList
import torch
from tqdm import tqdm
from transformers import BertTokenizer
import pandas as pd
from sklearn import metrics
import os
import gc
from sql_dao.sql_utils import get_conn, query_polarity_sql, delete_polarity_sql, insert_polarity
from pymysql.err import ProgrammingError, MySQLError
from ThreeSentiment.model import Model


LABEL_DICT = {0: '消极', 1: '中立', 2: '积极'}  # 标签映射表
SEQ_LENGTH = 128

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # 父目录


def load_model():
    PATH = parent_dir + '/model/roberta_three_sentiment_model.pth'
    TOKENIZER = BertTokenizer.from_pretrained(parent_dir + "/chinese_wwm_ext_pytorch")  # 模型[roberta-wwm-ext]所在的目录名称
    # 加载模型
    MODEL = Model(num_classes=3)
    DEVICE = torch.device("cpu")
    # DEVICE = torch.device("cpu")
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()
    MODEL.load_state_dict(torch.load(PATH, map_location=DEVICE), False)
    # print('模型加载完毕')
    return TOKENIZER, MODEL, DEVICE


#  批量进行情感分析 返回情感极性结果
def predict(sentences, MODEL, DEVICE):
    if type(sentences) == str:
        sentences = [sentences]
    dataloader = genDataLoaderFromList(sentences)

    preds = []

    for batch in dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            y_ = MODEL(**batch)
            pred = torch.argmax(y_, dim=1)
            preds.extend(pred.tolist())
    return preds


def predictWithLabel(sentence, MODEL, DEVICE):

    preds = predict(sentence, MODEL, DEVICE)

    labels = []

    for pred in preds:
        labels.append(LABEL_DICT[pred])

    return labels


#  利用sklearn计算模型的性能指标accuracy(准确率）、precision(精确率）、recall(召回率)、f1score(F1值)、auc
def model_all_target_test():
    TOKENIZER, MODEL, DEVICE = load_model()
    test_loader = genDataLoader(False)
    y_true = []
    y_pred = []
    y_score = []  # 概率值
    #  遍历所有行
    for (x1, x2, x3, y) in tqdm(test_loader):
        x1, x2, x3 = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE)
        y_true.extend(y.to(DEVICE).tolist())

        with torch.no_grad():
            y_ = MODEL(x1, token_type_ids=x2, attention_mask=x3)
            # probability = MODEL.forward(record['review'], token_type_ids=cur_type, attention_mask=cur_mask)  #计算得到概率值
            pred = torch.argmax(y_, dim=1)  # 预测的标签
        y_pred.extend(pred.tolist())
        # y_score.append(probability)

    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='macro')
    recall = metrics.recall_score(y_true, y_pred, average='macro')
    f1score = metrics.f1_score(y_true, y_pred, average='macro')
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
    preds = predict(comments, MODEL, DEVICE)
    positive = 0
    negative = 0
    neutrality = 0
    for pred in preds:
        if pred == 2:
            positive += 1
        elif pred == 0:
            negative += 1
        elif pred == 1:
            neutrality += 1

    # print(positive, negative, neutrality)

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
    polarity = predictWithLabel(['好精致的！特别适合妹妹！', "变身国企~~~店大欺客~~~珍爱生命~~~远离蒙牛~~~"], MODEL, DEVICE)
    print(polarity)


if __name__ == '__main__':
    my_test()

    # start_time = time.time()
    # polarity_analyze_by_workId(225)
    # print(time.time() - start_time)

    # accuracy, precision, recall, f1score, auc = model_all_target_test()

    # accuracy, precision, recall, f1score = model_all_target_test()
    # print(f"accuracy: {accuracy*100:.4f} "
    #       f"| precision: {precision*100:.4f} "
    #       f"| recall: {recall*100:.4f} | F1: {f1score*100:.4f}")
    pass
