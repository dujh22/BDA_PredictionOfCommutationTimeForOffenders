import pandas as pd
import torch
import jieba
import torchtext
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def cut(sentence):
    return sentence.split(' ')


TEXT = torchtext.data.Field(sequential=True, lower=False, tokenize=cut)
LABEL = torchtext.data.LabelField(sequential=False)



train_dataset,dev_dataset, test_dataset = torchtext.data.TabularDataset.splits(
    path="use_data",
    format='csv',
    skip_header=True,
    train="train.csv",
    validation ="dev.csv",
    test="test.csv",
    fields =[('text',TEXT),('label',LABEL)] 
)

#print(vars(train_dataset.examples[0])['text'])


pretrained_name = 'sgns.sogou.word'
pretrained_path = 'word_embedding'
vectors = torchtext.vocab.Vectors(name=pretrained_name, cache=pretrained_path)


TEXT.build_vocab(train_dataset, dev_dataset,test_dataset,vectors=vectors)
LABEL.build_vocab(train_dataset, dev_dataset,test_dataset)


train_iter, dev_iter,test_iter = torchtext.data.BucketIterator.splits(
        (train_dataset, dev_dataset,test_dataset),
        batch_sizes=(256, 128,128), 
        sort_key=lambda x: len(x.text) #按什么顺序来排列batch，这里是以句子的长度，就是上面说的把句子长度相近的放在同一个batch里面
        )


class TextCNN(nn.Module):
    def __init__(self,
                 filter_sizes, #卷积核长度
                 filter_num, #卷积核数量
                 vocabulary_size,
                 embedding_dimensions,
                 vectors
                 ):
        super(TextCNN, self).__init__()
        chanel_num = 1

        self.embedding = nn.Embedding(vocabulary_size, embedding_dimensions)
        self.embedding = self.embedding.from_pretrained(vectors)

        self.convs = nn.ModuleList(
            [nn.Conv2d(chanel_num, filter_num, (fsz, embedding_dimensions)) for fsz in filter_sizes]
        )

        self.fc1 = nn.Linear(len(filter_sizes) * filter_num, 36)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = x.unsqueeze(1)
        x = [conv(x) for conv in self.convs]
        x = [sub_x.squeeze(3) for sub_x in x]
        x = [F.relu(sub_x) for sub_x in x]
        x = [F.max_pool1d(sub_x, sub_x.size(2)) for sub_x in x]
        x = [sub_x.squeeze(2) for sub_x in x]
        x = torch.cat(x, 1)
        logits = self.fc1(x)
        res = F.log_softmax(logits,dim=1)
        return res


filter_sizes = [2,3,4]
filter_num = 32
vocab_size = len(TEXT.vocab)
embedding_dim = TEXT.vocab.vectors.size()[-1]
vectors = TEXT.vocab.vectors
lr = 0.005
epochs = 20


model = TextCNN(
    filter_sizes=filter_sizes,
    filter_num=filter_num,
    vocabulary_size=vocab_size,
    embedding_dimensions=embedding_dim,
    vectors=vectors)

model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
loss = nn.CrossEntropyLoss()
for epoch in range(epochs):
    train_l = 0
    model.train()
    steps = 0
    for batch in train_iter:
        steps += 1
        text, label = batch.text, batch.label
        text, label = text.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(text)
        l = loss(output, label)
        train_l += l
        l.backward()
        optimizer.step()

    model.eval()
    resnum = 0
    num = 0
    for batch in dev_iter:
        text, label = batch.text, batch.label
        text, label = text.cuda(), label.cuda()
        with torch.no_grad():
            output = model(text)
        result = np.argmax(output.cpu().numpy(),axis=1)
        res = result - label.cpu().numpy()
        resnum += np.square(res).sum()
        num += batch.batch_size 

    print("epoch:{}, loss: {} , MSE: {}".format(epoch, train_l / steps , math.sqrt(resnum/ num)))


def getv(lp,la):
    return np.abs(np.log(lp+1) - np.log(la + 1))


def getscore(v):
    if v <= 0.2:
        return 1
    elif v<=0.4:
        return 0.8
    elif v<=0.6:
        return 0.6
    elif v<=0.8:
        return 0.4
    elif v <= 1:
        return 0.2
    else:
        return 0


num = 0
resnum = 0
score = 0
model.eval()
#for batch in dev_iter:
for batch in test_iter:
    text, label = batch.text,batch.label
    text, label = text.cuda(), label.cuda()
    with torch.no_grad():
      output = model(text)
    result = np.argmax(output.cpu().numpy(),axis=1)
    resnum += np.sum(result == label.cpu().numpy())
    for lp,la in zip(result,label.cpu()):
      score += getscore(getv(lp, la))
    num += batch.batch_size
ext = resnum / num
score = score / num
finalscore = score * 0.7 + ext * 0.3
print("score:{}, extacc: {}, finalscore:{}".format(score, ext, finalscore))
#0.5530576330840261 0.12032556093268808 0.42323801143862466 回归

#0.5450505939287278 0.27188737351517817 0.46310162780466285 分类