import math
import matplotlib.pyplot as plt
import re
import tqdm
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertConfig, BertTokenizer, BertForMaskedLM, AdamW, get_cosine_schedule_with_warmup

#drive_dir = "/content/drive/MyDrive/my_hw/"
drive_dir = "./"
pretrained = "bert-base-chinese"
SEQ_LEN = 512

def preprocess():
    x, y = [], []
    with open(drive_dir + "train.csv", "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
        cut_words = ["服刑", "户籍", "文化", "族", "生", "男", "女"]    # 用于去除最前面的罪犯个人信息
        cut_words_len = len(cut_words) - 1
        index = 0
        count = 0
        for line in lines:
            line_info = line.split(",")
            text = line_info[1]
            name = text[2:text.find("，")]  # 去除人名
            if len(name) > 1:
                text = re.sub("%s+" % name, "", text)
            text = text[text.find(" ") + 1:]    # 去除第一个空格前的所有内容（可以改为只保留犯xx罪，刑期xx年）
            cut_pos = text.find(cut_words[index])   # 去除罪犯个人信息，其实通过上一步的处理这里有点多余
            while cut_pos == -1 and index < cut_words_len:
                index += 1
                cut_pos = text.find(cut_words[index])
            index = 0
            if cut_pos != -1:
                cut_pos = text.find("，", cut_pos)
                text = text[cut_pos + 1 :]
            text = re.sub(u"\\（.*?\\）|\\{.*?}|\\［.*?］", '', text)   # 去除括号内的内容
            text = re.sub("[：+——?【】《》“”！，。？、~@#￥%……&*（）]+", "", text)   # 去除标点符号等字符
            if len(text) > SEQ_LEN: # 最大长度为512
                text = text[-SEQ_LEN: ]
            if count % 10 == 0 and count < 100:
                print(text)
            count += 1
            x.append(text)
            y.append(int(float(line_info[2])))

    tokenizer = BertTokenizer.from_pretrained(pretrained)
    input_ids, attention_masks, input_types = [], [], []
    for text in x:
        input = tokenizer(text, max_length=SEQ_LEN, padding="max_length", truncation=True)
        input_ids.append(input["input_ids"])
        attention_masks.append(input["attention_mask"])
        input_types.append(input["token_type_ids"])

    input_ids, attention_masks, input_types = np.array(input_ids), np.array(attention_masks), np.array(input_types)
    y = np.array(y)
    print("shape: ", input_ids.shape, attention_masks.shape, input_types.shape, y.shape)
    np.save(drive_dir + "preprocess_data/train_ids.npy", input_ids)
    np.save(drive_dir + "preprocess_data/train_masks.npy", attention_masks)
    np.save(drive_dir + "preprocess_data/train_types.npy", input_types)
    np.save(drive_dir + "preprocess_data/train_y.npy", y)

preprocess()
input_ids = np.load(drive_dir + "preprocess_data/train_ids.npy")
attention_masks = np.load(drive_dir + "preprocess_data/train_masks.npy")
input_types = np.load(drive_dir + "preprocess_data/train_types.npy")
y = np.load(drive_dir + "preprocess_data/train_y.npy")
print("shape: ", input_ids.shape, attention_masks.shape, input_types.shape, y.shape)

length = input_ids.shape[0]
train_size = length // 10 * 8
valid_size = length // 10 * 1
indexes = np.arange(length)
np.random.seed(1234)
np.random.shuffle(indexes)
input_ids_train, input_ids_val, input_ids_test = input_ids[indexes[:train_size]], input_ids[indexes[train_size: train_size + valid_size]], input_ids[indexes[train_size + valid_size: ]],
attention_masks_train, attention_masks_val, attention_masks_test = attention_masks[indexes[:train_size]], attention_masks[indexes[train_size: train_size + valid_size]], attention_masks[indexes[train_size + valid_size: ]]
input_types_train, input_types_val, input_types_test = input_types[indexes[:train_size]], input_types[indexes[train_size: train_size + valid_size]], input_types[indexes[train_size + valid_size:]]
y_train, y_val, y_test = y[indexes[:train_size]], y[indexes[train_size: train_size + valid_size]], y[indexes[train_size + valid_size: ]]

BATCH_SIZE = 16

train_data = TensorDataset(torch.LongTensor(input_ids_train), torch.LongTensor(attention_masks_train), torch.LongTensor(input_types_train), torch.LongTensor(y_train))
train_sampler = RandomSampler(train_data)
train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

valid_data = TensorDataset(torch.LongTensor(input_ids_val), torch.LongTensor(attention_masks_val), torch.LongTensor(input_types_val), torch.LongTensor(y_val))
valid_sampler = SequentialSampler(valid_data)
valid_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(torch.LongTensor(input_ids_test), torch.LongTensor(attention_masks_test), torch.LongTensor(input_types_test))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

print(len(train_data), len(train_loader))
print(len(valid_data), len(valid_loader))
print(len(test_data), len(test_loader))

class Bert(nn.Module):
    def __init__(self, classes=36, dropout=0.5):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.config = BertConfig.from_pretrained(pretrained)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.config.hidden_size, classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        output = self.bert(input_ids, attention_mask, token_type_ids)
        output = output[1]
        # output = self.dropout(output)    # modified
        # print(output)
        output = self.fc(output)
        # output = nn.ReLU(output)
        output = F.log_softmax(output, dim=1)
        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 5
model = Bert().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader), num_training_steps=epochs * len(train_loader))

def cal_test(label, pred):
    acc = np.mean(label == pred)
    score1 = np.abs(np.log(pred + 1) - np.log(label + 1))
    for i in range(len(score1)):
        if score1[i] <= 0.2:
            score1[i] = 1
        elif score1[i] > 0.2 and score1[i] <= 0.4:
            score1[i] = 0.8
        elif score1[i] > 0.4 and score1[i] <= 0.6:
            score1[i] = 0.6
        elif score1[i] > 0.6 and score1[i] <= 0.8:
            score1[i] = 0.4
        elif score1[i] > 0.8 and score1[i] <= 1:
            score1[i] = 0.2
        else:
            score1[i] = 0
    mae = np.mean(np.abs(label - pred))
    rmse = math.sqrt(np.mean(np.square(label - pred)))
    final_score = acc * 0.3 + np.mean(score1) * 0.7
    print("Acc:", acc, "score1: ", np.mean(score1), "FinalScore: ", final_score, "MAE:", mae, "RMSE:", rmse)
    return final_score

def evaluate(model, data_loader, device):
    model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for idx, att, type, y in data_loader:
            y_pred = model(idx.to(device), att.to(device), type.to(device))
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
            val_true.extend(y.squeeze().cpu().numpy().tolist())
    final_score = cal_test(np.array(val_true), np.array(val_pred))
    # return accuracy_score(val_true, val_pred)
    return final_score, accuracy_score(val_true, val_pred)

def predict(model, data_loader, device):    # 测试集
    model.eval()
    test_pred = []
    with torch.no_grad():
        for idx, att, type in tqdm(data_loader):
            y_pred = model(idx.to(device), att.to(device), type.to(device))
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            test_pred.extend(y_pred)
    return test_pred

def train_and_eval(model, train_loader, valid_loader, optimizer, scheduler, device, epoch):
    best_score = 0.0
    best_acc = 0.0
    criterion = nn.CrossEntropyLoss().to(device)
    for i in range(epoch):
        torch.cuda.empty_cache()
        start = time.time()
        model.train()
        print("Epoch {}".format(i + 1))
        train_loss = 0.0
        for idx, (ids, att, tpe, y) in enumerate(train_loader):
            ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(device), y.to(device)
            y_pred = model(ids, att, tpe)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            if (idx + 1) % (len(train_loader) // 50) == 0:
                print("Epoch {:02d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}s".format(i + 1, idx + 1, len(train_loader), train_loss / (idx + 1), time.time() - start))

        model.eval()
        final_score, exact_acc = evaluate(model, valid_loader, device)
        if exact_acc > best_acc:
            best_acc = exact_acc
        if final_score > best_score:
            best_score = final_score
            torch.save(model.state_dict(), drive_dir + "best_bert_model.pth")

        print("current final_score is {:.6f}, best final_score is {:.6f}".format(final_score, best_score))
        print("current acc is {:.4f}, best acc is {:.4f}".format(exact_acc, best_acc))


train_and_eval(model, train_loader, valid_loader, optimizer, scheduler, device, epochs)