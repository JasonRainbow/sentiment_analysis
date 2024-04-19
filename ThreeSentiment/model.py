import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import math


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained('../chinese_wwm_ext_pytorch')  # /roberta-wwm-ext pretrain/
        for param in self.bert.parameters():
            param.requires_grad = True  # 所有参数求梯度

        # 定义超参数
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100
        bert_hidden_size = self.bert.config.hidden_size

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, bert_hidden_size)) for K in self.filter_sizes]
        )

        # LSTM
        self.lstm = nn.LSTM(input_size=bert_hidden_size,
                            hidden_size=320, num_layers=1, batch_first=True)

        # Self-Attention
        self.key_layer = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.query_layer = nn.Linear(bert_hidden_size, bert_hidden_size)
        self.value_layer = nn.Linear(bert_hidden_size, bert_hidden_size)
        self._norm_fact = 1 / math.sqrt(bert_hidden_size)

        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(620, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
        )

        self.fc = nn.Linear(in_features=768, out_features=num_classes)
        self.dropout = nn.Dropout(0.5)

    def conv_pool(self, tokens, conv):
        # x -> [batch,  1, text_length, 768]
        tokens = conv(tokens)  # shape [batch_size, out_channels, x.shape[2]] - conv.kernel_size[0]+1, 1]
        tokens = F.relu(tokens)
        tokens = tokens.squeeze(3)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0]+1]
        tokens = F.max_pool1d(tokens, tokens.size(2))  # shape [batch_size, out_channels, 1]
        out = tokens.squeeze(2)  # shape [batch_size, out_channels]

        return out

    def forward(self, input_ids, token_type_ids, attention_mask):
        context = input_ids  # 输入的句子
        types = token_type_ids
        mask = attention_mask  # 对padding部分进行mask，和句子相同size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        output = self.bert(context, token_type_ids=types, attention_mask=mask)

        # out = self.fc(output[0][:, 0])  # 得到三分类概率
        # out = self.dropout(out)
        # return out

        tokens = output.last_hidden_state

        # Self-Attention
        K = self.key_layer(tokens)
        Q = self.query_layer(tokens)
        V = self.value_layer(tokens)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        attention_output = torch.bmm(attention, V)

        # TextCNN
        cnn_tokens = attention_output.unsqueeze(1)  # shape [batch_size, 1, max_len, hidden_size]
        cnn_out = torch.cat([self.conv_pool(cnn_tokens, conv) for conv in self.convs],
                            1)  # shape [batch_size, self.num_filters * len(self.filter_sizes)]

        rnn_tokens = tokens
        rnn_outputs, _ = self.lstm(rnn_tokens)
        rnn_out = rnn_outputs[:, -1, :]
        # cnn_out --> [batch, 300]
        # rnn_out --> [batch, 512]
        out = torch.cat((cnn_out, rnn_out), 1)
        predicts = self.block(out)

        # predicts = self.block(output[0][:, 0])

        return predicts
