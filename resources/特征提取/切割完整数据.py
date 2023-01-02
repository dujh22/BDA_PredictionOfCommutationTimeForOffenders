# _*_coding:utf-8 _*_

import shutil
import os

shutil.rmtree("单个案件")
os.mkdir("单个案件")

# 将整个文本切割成单个的案件
with open("./result.txt", "r", encoding="utf-8") as f1:
    cases = f1.readlines()
    num = 1
    file_name = '单个案件/' + str(num) + '.txt'
    for i in range(len(cases)):
        f2 = open(file_name, "a", encoding='utf-8')
        if cases[i] == '\n':
            num += 1
            file_name = '单个案件/' + str(num) + '.txt'
        else:
            f2.writelines(cases[i])
        f2.close()
