import gc

from SixSentiment.build_data import convert_text_to_token
from SixSentiment.model import Model, pretrained_model_path
import torch
from transformers import BertTokenizer
import numpy as np
import pandas as pd
from sql_dao.sql_utils import get_conn, insert_sentiment, query_sentiment_sql, delete_sentiment_sql
from pymysql.err import ProgrammingError, MySQLError
import os


LABEL_DICT = {0: '恐惧', 1: '中立', 2: '伤心', 3: '惊讶', 4: '愤怒', 5: '开心'}  # 标签映射表
SEQ_LENGTH = 128

parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # 父目录


def load_model():
    PATH = parent_dir + '/model/roberta_model.pth'
    TOKENIZER = BertTokenizer.from_pretrained(pretrained_model_path)  # 模型[roberta-wwm-ext]所在的目录名称
    # 加载模型
    MODEL = Model(num_classes=6)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = MODEL.to(DEVICE)
    MODEL.load_state_dict(torch.load(PATH, map_location=DEVICE), False)
    MODEL.eval()
    # print('模型加载完毕')
    return TOKENIZER, MODEL, DEVICE


def predict_single_sentence(sentence, TOKENIZER, MODEL, DEVICE):
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
        cur_pre = LABEL_DICT[int(pred[0][0].data.cpu().numpy())]  # 预测的情绪
    return cur_pre


def sentiment_analyze_by_workId(workId):
    print("workId：%d ，开始进行细腻情感分析" % workId)
    TOKENIZER, MODEL, DEVICE = load_model()
    conn = get_conn()
    sql1 = "select distinct country, platform, postTime from raw_comment where workId = %d" % workId
    # 分别获取workId号作品的评论所属的国家列表、平台列表、发布时间列表
    df = pd.read_sql(sql1, con=conn)
    for i in range(len(df)):
        sentiment_analyze(workId, df["country"][i], df["platform"][i],
                          df["postTime"][i], TOKENIZER, MODEL, DEVICE, conn)
    conn.close()
    del TOKENIZER, MODEL, DEVICE, conn
    gc.collect()
    print("workId：%d ，完成细腻情感分析" % workId)
    return True


def sentiment_analyze(workId, country, platform, post_time, TOKENIZER, MODEL, DEVICE, conn):
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
    happy = 0
    amazed = 0
    neutrality = 0
    sad = 0
    angry = 0
    fear = 0
    for comment in comments:
        if comment.strip() == 0:
            continue
        res = predict_single_sentence(comment, TOKENIZER, MODEL, DEVICE)
        if res == "恐惧":
            fear += 1
        elif res == "中立":
            neutrality += 1
        elif res == "伤心":
            sad += 1
        elif res == "惊讶":
            amazed += 1
        elif res == "愤怒":
            angry += 1
        elif res == "开心":
            happy += 1

    # print(comments)
    cursor = conn.cursor()
    cursor.execute(query_sentiment_sql.format(workId, country, platform, post_time))
    if len(cursor.fetchall()) > 0:
        cursor.execute(delete_sentiment_sql.format(workId, country, platform, post_time))
    try:
        # print("写入数据库")
        insert_sentiment(workId, country, platform, post_time, happy, amazed, neutrality, sad, angry, fear, conn)
    except (MySQLError, ProgrammingError):
        print("插入失败，有错误")
    finally:
        conn.commit()  # 提交修改


def my_test():
    TOKENIZER, MODEL, DEVICE = load_model()
    print(predict_single_sentence("我喜欢这部电影", TOKENIZER, MODEL, DEVICE))


if __name__ == '__main__':
    my_test()
    # print(load_model()[1].eval())
    # sentiment_analyze_by_workId(19)  # 六分类情感分析
    pass
