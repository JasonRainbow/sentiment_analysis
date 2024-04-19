import torch
import torch.nn as nn
from transformers import BertModel
import os


pretrained_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) + '/chinese_wwm_ext_pytorch'


# 复用模型结构
class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path)  # /roberta-wwm-ext pretrain/
        for param in self.bert.parameters():
            param.requires_grad = True  # 所有参数求梯度
        # self.fc = nn.Linear(768, num_classes)  # 768 -> 6
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
        )

    def forward(self, x, token_type_ids, attention_mask):
        context = x  # 输入的句子
        types = token_type_ids
        mask = attention_mask  # 对padding部分进行mask，和句子相同size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        output_1 = self.bert(context, token_type_ids=types, attention_mask=mask)
        hidden_state = output_1[0]
        pooled = hidden_state[:, 0]
        # out = self.fc(output_1[1])  # 得到6分类概率
        out = self.block(pooled)
        return out


if __name__ == "__main__":
    print()
