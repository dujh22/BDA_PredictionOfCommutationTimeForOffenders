from tqdm import tqdm
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertModel, BertConfig, BertTokenizer

drive_dir = "/content/drive/MyDrive/my_hw/"
pretrained = "bert-base-chinese"
SEQ_LEN = 512

def preprocess():
    x = []
    with open(drive_dir + "train.csv", "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]
        cut_words = ["服刑", "户籍", "文化", "族", "生", "男", "女"]
        cut_words_len = len(cut_words) - 1
        index = 0
        for line in lines:
            line_info = line.split(",")
            text = line_info[1]
            name = text[2:text.find("，")]
            if len(name) > 1:
                text = re.sub("%s+" % name, "", text)
            text = text[text.find(" ") + 1:]
            cut_pos = text.find(cut_words[index])
            while cut_pos == -1 and index < cut_words_len:
                index += 1
                cut_pos = text.find(cut_words[index])
            index = 0
            if cut_pos != -1:
                cut_pos = text.find("，", cut_pos)
                text = text[cut_pos + 1 :]
            text = re.sub(u"\\（.*?\\）|\\{.*?}|\\［.*?］", '', text)
            text = re.sub("[：+——?【】《》“”！，。？、~@#￥%……&*（）]+", "", text)
            if len(text) > 512:
                text = text[-512: ]
            x.append(text)

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
    np.save(drive_dir + "preprocess_data/test_ids.npy", input_ids)
    np.save(drive_dir + "preprocess_data/test_masks.npy", attention_masks)
    np.save(drive_dir + "preprocess_data/test_types.npy", input_types)

preprocess()

input_ids = np.load(drive_dir + "preprocess_data/test_ids.npy")
attention_masks = np.load(drive_dir + "preprocess_data/test_masks.npy")
input_types = np.load(drive_dir + "preprocess_data/test_types.npy")

BATCH_SIZE = 32

test_data = TensorDataset(torch.LongTensor(input_ids), torch.LongTensor(attention_masks), torch.LongTensor(input_types))
test_sampler = SequentialSampler(test_data)
test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)


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
model = Bert().to(device)


def predict(model, data_loader, device):
    model.eval()
    test_pred = []
    with torch.no_grad():
        for idx, (ids, att, tpe) in tqdm(enumerate(data_loader)):
            y_pred = model(ids.to(device), att.to(device), tpe.to(device))
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            test_pred.extend(y_pred)
    return test_pred


model.load_state_dict(torch.load(drive_dir + "best_bert_model.pth"))
predicts = predict(model, test_loader, device)
with open(drive_dir + "predict.csv", "w", encoding="utf-8") as f:
    lines = ["Id,Predicted\n"]
    for i in range(1, 25001 + 1):
        lines.append("{},{}\n".format(i, float(predicts[i - 1])))
    f.writelines(lines)