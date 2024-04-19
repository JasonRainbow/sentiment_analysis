import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertTokenizerFast, BertTokenizer
import pandas as pd
from ThreeSentiment.model import Model
from ThreeSentiment.build_data import genDataLoader


batch_size = 32
num_classes = 3
max_length = 128
model_pretrained_path = "../chinese_wwm_ext_pytorch"
model_finetune_path = "../model/roberta_three_sentiment_model.pth"


class CommentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx])
                for k, v in self.encodings.items()}

        return item, self.labels[idx]

    def __len__(self):
        return len(self.labels)


def load_data(tokenizer: BertTokenizerFast, file_path):
    df = pd.read_csv(file_path, sep="\t", encoding="utf-8")
    encodings = tokenizer(df["review"].values.tolist(),
                          truncation=True,
                          padding="max_length",
                          return_token_type_ids=True,
                          max_length=max_length)
    dataset = CommentDataset(encodings, df["label"].values.tolist())

    return DataLoader(dataset, batch_size=batch_size)


# 计算模型在验证集上的性能
def valid_model(model, dataloader, DEVICE):
    model.eval()
    valid_loss = 0
    acc = 0

    for (inputs, targets) in tqdm(dataloader):
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        targets = targets.to(DEVICE)
        with torch.no_grad():
            outputs = model(inputs["input_ids"],
                            token_type_ids=inputs["token_type_ids"],
                            attention_mask=inputs["attention_mask"])
            valid_loss += F.cross_entropy(outputs, targets)
            acc += (torch.argmax(outputs, dim=1) == targets).sum()

    valid_loss /= len(dataloader)

    print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        valid_loss, acc, len(dataloader.dataset),
        100. * acc / len(dataloader.dataset)))

    return acc / len(dataloader.dataset)


def test_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(num_classes)
    model.load_state_dict(torch.load(model_finetune_path), False)
    model.eval()
    model.to(DEVICE)
    tokenizer = BertTokenizerFast.from_pretrained(model_pretrained_path)
    # tokenizer2 = BertTokenizer.from_pretrained(model_pretrained_path)
    test_dl = load_data(tokenizer, "../data/three_sentiment_test.csv")
    # test_dl = genDataLoader(False)
    valid_model(model, test_dl, DEVICE)


if __name__ == "__main__":
    test_model()
