import os
import random
import re
import math
import time
import jieba

DATA_PATH = '../jyxstxtqj/'


def get_single_corpus(file_path):
    """
    获取file_path文件对应的内容
    :return: file_path文件处理结果
    """
    corpus = ''
    # unuseful items filter
    r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~「」『』（）]+'
    with open('../stopwords.txt', 'r', encoding='utf8') as f:
        stop_words = [word.strip('\n') for word in f.readlines()]
        f.close()
    # print(stop_words)
    with open(file_path, 'r', encoding='ANSI') as f:
        corpus = f.read()
        corpus = re.sub(r1, '', corpus)
        corpus = corpus.replace('\n', '')
        corpus = corpus.replace('\u3000', '')
        corpus = corpus.replace('本书来自免费小说下载站更多更新免费电子书请关注', '')
        f.close()
    words = list(jieba.cut(corpus))
    return [word for word in words if word not in stop_words]


def get_segment(index_file, length):
    """
    随机获得长度为length的选段
    :param index_file: 供选取的小说列表
    :param length: 选段的长度
    :return: 保存的选段路径，选段内容
    """
    segment = []
    with open(index_file, 'r') as f:
        txt_list = f.readline().split(',')
        file_path = txt_list[random.randint(0, len(txt_list)-1)] + '.txt'
        split_words = get_single_corpus(DATA_PATH + file_path)
        start = random.randint(0, len(split_words)-length-1)
        segment.extend(split_words[start: start+length])
    return file_path, segment


if __name__ == '__main__':
    # calculate_inf_entropy('inf.txt')
    for i in range(200):
        temp_path, temp_seg = get_segment('../inf2.txt', 1000)
        with open('../segments2/' + str(i) + temp_path, 'w', encoding='utf8') as f:
            for word in temp_seg:
                f.write(word + '\n')
        print(temp_path)
        # print(temp_seg[0:10])


