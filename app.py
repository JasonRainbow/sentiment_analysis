from flask import Flask, jsonify, request
from SixSentiment.detailed_emotion_analysis import sentiment_analyze_by_workId
from ThreeSentiment.BertModelTest import polarity_analyze_by_workId
from subject_analysis.subject_extract import batch_extract, predict_classify

app = Flask(__name__)


def success(data=None):
    json_res = {
        "code": "0",
        "msg": "响应成功",
        "data": data
    }
    return jsonify(json_res)


def err_res(msg, code=-1):
    json_res = {
        "code": str(code),
        "msg": msg,
        "data": None
    }
    return jsonify(json_res)


@app.route('/')
def hello_world():  # put application's code here
    return success()


@app.route('/analyze_polarity', methods=["GET"])
def analyze_polarity():  # 情感极性分析
    args = request.args
    workId = args.get("workId", default=0, type=int)
    if workId == 0:
        return err_res("请输入作品ID")
    res = polarity_analyze_by_workId(workId)
    if res:
        return success()
    else:
        return err_res("极性情感分析失败")


@app.route('/analyze_sentiment', methods=["GET"])
def analyze_sentiment():  # 细腻情感分析
    args = request.args
    workId = args.get("workId", default=0, type=int)
    if workId == 0:
        return err_res("请输入作品ID")
    res = sentiment_analyze_by_workId(workId)
    if res:
        return success()
    else:
        return err_res("细腻情感分析失败")


if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=5050)
    # batch_extract(task_path='./subject_analysis/checkpoint/model_best')
    res = predict_classify("特效: 非常好")
    print(res)
