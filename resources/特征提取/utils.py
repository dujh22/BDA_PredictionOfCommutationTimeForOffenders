import re

def remove_duplicate_elements(l):
    new_list = []
    for elem in l:
        if elem not in new_list:
            new_list.append(elem)
    return new_list


def find_element(l, *ss):
    for s in ss:
        for element in l:
            if s in element:
                return "1"
    return "0"


def text2num(text):
    num = 0
    # 将text序列连接成字符串
    text = "".join(text)
    digit = {
        '一': 1,
        '二': 2,
        '两': 2,
        '三': 3,
        '四': 4,
        '五': 5,
        '六': 6,
        '七': 7,
        '八': 8,
        '九': 9}
    if text:
        idx_q, idx_b, idx_s = text.find('千'), text.find('百'), text.find('十')
        if idx_q != -1:
            num += digit[text[idx_q - 1:idx_q]] * 1000
        if idx_b != -1:
            num += digit[text[idx_b - 1:idx_b]] * 100
        if idx_s != -1:
            # 十前忽略一的处理
            num += digit.get(text[idx_s - 1:idx_s], 1) * 10
        if text[-1] in digit:
            num += digit[text[-1]]
    return num


def per_num(text):
    string = re.findall(r"\d+", text)
    if len(string) == 0:
        r1 = re.compile(u'[一二两三四五六七八九十]{1,}')
        r2 = r1.findall(text)
        if len(r2) == 0:
            num = 1
        else:
            num = text2num(r2)
    else:
        num = string[0]
    return num


def extract_seg(content):
    # 死亡人数、重伤人数、轻伤人数提取
    r1 = re.compile(u'[1234567890一二两三四五六七八九十 ]*人( )*死亡')
    r2 = re.search(r1, content)
    if r2 is None:
        num1 = 0
    else:
        text = r2.group()
        num1 = per_num(text)
    # 重伤人数
    r3 = re.compile(u'[1234567890一二两三四五六七八九十 ]*人( )*重伤')
    r4 = re.search(r3, content)
    if r4 is None:
        num2 = 0
    else:
        text = r4.group()
        num2 = per_num(text)
    # 受伤人数
    r5 = re.compile(u'[1234567890一二两三四五六七八九十 ]*人( )*(轻伤|受伤)')
    r6 = re.search(r5, content)
    if r6 is None:
        num3 = 0
    else:
        text = r6.group()
        num3 = per_num(text)
    return num1, num2, num3

# 提取出判决结果，单位为月份
def sentence_result(text):
    text = text.strip(" ")  # 去除每行首尾可能出现的多余空格
    text = text.replace(" ", "")  # 去除所有空格
    if text.find("判决如下") != -1:
        result = text.split('判决如下')[-1]
    elif text.find("判处如下") != -1:
        result = text.split('判处如下')[-1]
    else:
        result = text
    r1 = re.compile(u'(有期徒刑|拘役)[一二三四五六七八九十又年零两]{1,}(个月|年)')
    r2 = re.search(r1, result)
    if r2 is None:
        num = 0
    else:
        text = r2.group()
        r3 = re.compile(u'[一二三四五六七八九十两]{1,}')
        r4 = r3.findall(text)
        if len(r4) > 1:
            num1 = text2num(r4[0])
            num2 = text2num(r4[1])
            num = 12 * num1 + num2
        elif text.find(u"年") != -1:
            num = 12 * text2num(r4)
        else:
            num = text2num(r4)
    return num
