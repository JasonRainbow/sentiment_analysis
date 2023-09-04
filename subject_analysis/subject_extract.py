from paddlenlp import Taskflow
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
import paddle
import functools
import os
import paddle.nn.functional as F
from paddle.io import BatchSampler, DataLoader
from text_classifier.utils import preprocess_function

parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # 父目录


# 提取属性词和观点词，并进行拼接
def extract_prop_and_op(text, predict):
    results = predict(text)[0]
    if results.get('评价维度') is None:
        return []
    results = results['评价维度']
    prop_op_lst = []
    for res in results:
        prop = res['text']
        polarity = "中立"
        if res.get('relations') is not None:
            if res.get('relations').get('观点词') is not None:
                relations = res['relations']['观点词']
                for (idx, relation) in enumerate(relations):
                    if idx == 0:
                        prop += ': '
                    else:
                        prop += ';'
                    prop += relation['text']
            if res.get('relations').get('情感倾向[积极,消极,中立]') is not None:
                polarity = res.get('relations').get('情感倾向[积极,消极,中立]')[0].get('text')

        prop_op_lst.append((prop, polarity))
    return prop_op_lst


# 对单条语句进行主题分类
def predict_classify(sentence):
    """
    Predicts the data labels.
    """
    paddle.set_device("gpu")
    model = AutoModelForSequenceClassification.from_pretrained(parent_dir + "/text_classifier/checkpoint/")
    tokenizer = AutoTokenizer.from_pretrained(parent_dir + "/text_classifier/checkpoint/")

    label_list = []
    label_path = os.path.join(parent_dir + "/text_classifier/data", "label.txt")
    with open(label_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            label_list.append(line.strip())

    data_ds = MapDataset([{"sentence": sentence}])

    trans_func = functools.partial(
        preprocess_function,
        tokenizer=tokenizer,
        max_seq_length=128,
        label_nums=len(label_list),
        is_test=True,
    )

    data_ds = data_ds.map(trans_func)

    # batchify dataset
    collate_fn = DataCollatorWithPadding(tokenizer)
    data_batch_sampler = BatchSampler(data_ds, batch_size=16, shuffle=False)

    data_data_loader = DataLoader(dataset=data_ds, batch_sampler=data_batch_sampler, collate_fn=collate_fn)

    results = []
    model.eval()
    for batch in data_data_loader:
        logits = model(**batch)
        probs = F.sigmoid(logits).numpy()
        for prob in probs:
            labels = []
            for i, p in enumerate(prob):
                if p > 0.5:
                    labels.append(i)
            results.append(labels)
    predict_labels = []
    for d, result in zip(data_ds.data, results):
        label = [label_list[r] for r in result]
        predict_labels.append(",".join(label))
    return predict_labels


def batch_extract(task_path='./checkpoint/model_best'):
    # 使用训练后的模型进行预测
    schema = [{'评价维度': ['观点词', '情感倾向[积极,消极,中立]']}]  # 模式
    senta = Taskflow("sentiment_analysis", model='uie-senta-base',
                     schema=schema, task_path=task_path)
    res = extract_prop_and_op('这部电影的特效非常好', senta)
    print(res)


if __name__ == "__main__":
    # batch_extract()
    res = predict_classify("特效: 非常好")
    print(res)
