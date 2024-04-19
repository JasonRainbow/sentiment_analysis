import torch
from SixSentiment.build_data import genDataLoader
from tqdm import tqdm  # 注意不要直接 import tqdm
import torch.nn.functional as F
from torch.optim import lr_scheduler
from SixSentiment.model import Model


# 加载模型
MODEL1 = Model(num_classes=6)  # 指定分类类别
print('原始模型加载完毕')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = MODEL1.to(DEVICE)  # 将模型分配到CPU
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=1e-5)  # 优化器
NUM_EPOCHS = 6  # epoch
PATH = '../model/roberta_model.pth'  # 定义模型保存路径


def train(model, device, train_loader, test_loader, optimizer):  # 训练模型
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):  # 3个epoch
        model.train()  # 每个 batch 独立计算其均值和方差
        batch_idx = 0
        for (x1, x2, x3, y) in tqdm(train_loader):
            x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
            y_pred = model(x1, token_type_ids=x2, attention_mask=x3)  # 得到预测结果
            optimizer.zero_grad()  # 梯度清零
            loss = F.cross_entropy(y_pred, y)  # 得到loss
            # accu_loss += loss.item() # 计算累积loss
            loss.backward()
            optimizer.step()
            batch_idx += 1
            if (batch_idx + 1) % 100 == 0:  # 打印loss
                print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx + 1) * len(x1),
                                                                               len(train_loader.dataset),
                                                                               100. * batch_idx / len(train_loader),
                                                                               # accu_loss / batch_idx))
                                                                               loss.item()))  # 记得为loss.item()
        acc = model_test(model, device, test_loader)  # 每个epoch结束后评估一次测试集精度
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), PATH)  # 保存最优模型

        # 根据训练情况调整学习率
        exp_lr_scheduler.step()


def model_test(model, device, test_loader):  # 测试模型, 得到测试集评估结果
    model.eval()  # 用之前统计的值来测试
    test_loss = 0.0
    acc = 0
    for (x1, x2, x3, y) in tqdm(test_loader):
        x1, x2, x3, y = x1.to(device), x2.to(device), x3.to(device), y.to(device)
        with torch.no_grad():
            y_ = model(x1, token_type_ids=x2, attention_mask=x3)
        test_loss += F.cross_entropy(y_, y)
        pred = y_.max(-1, keepdim=True)[1]  # .max(): 2输出，分别为最大值和最大值的index
        acc += pred.eq(y.view_as(pred)).sum().item()  # 记得加item()
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, acc, len(test_loader.dataset),
        100. * acc / len(test_loader.dataset)))
    return acc / len(test_loader.dataset)


def main():
    train_data = genDataLoader(True)
    print('训练集处理完毕')
    test_data = genDataLoader(False)
    print('测试集处理完毕')
    train(MODEL, DEVICE, train_data, test_data, OPTIMIZER)

    # MODEL.load_state_dict(torch.load(PATH), False)
    # model_test(MODEL, DEVICE, test_data)


if __name__ == '__main__':
    main()
