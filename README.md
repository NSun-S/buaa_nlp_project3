## 深度学习与自然语言处理第二次作业

<p align='right'> <strong>SY2106318 孙旭东 </strong></p>

**详细技术报告见**[SY2106318-深度学习和自然语言处理第三次作业](https://github.com/NSun-S/buaa_nlp_project3/raw/main/SY2106318-深度学习和自然语言处理第三次作业.pdf)

### 程序简介

#### 程序运行

代码文件共三个:

- data_prepare.py：生成随机选段
- main.py：LDA模型的生成，生成片段对应的特征并保存
- svm.py：需要参数--mode，为'train'或'test'，用于生成svm模型，测试模型效果。

#### 主要模块介绍

首先定义模型需要使用的一些变量及参数：

```python
word2id = {}  # word和id对应map
id2word = {}  # id和word对应map
word2topic = []  # 每个word所属主题
topic_num = 20  # 主题数
file_num = len(segments)  # 文档数
word_num = len(word2id)  # 总词数
alpha = len(segments[0])/topic_num/10  # 初始alpha
beta = 1/topic_num  # 初始beta
file2topic = np.zeros([file_num, topic_num]) + alpha  # 文档各主题分布
topic2word = np.zeros([topic_num, word_num]) + beta  # 主题各词分布
topic_count = np.zeros([topic_num]) + word_num * beta  # 每个主题总词数
```

LDA模型的核心部分包括模型初始化和Gibbs Sampling两部分，在模型初始化时，随机为文档中的每个词分配一个主题，之后统计每个主题$z$下词出现的概率，及每个文档$m$下出现主题$z$的数量，以及每个主题下词的总量，初始化代码如下：

```python
def initialize():
    for f_idx, segment in enumerate(segments):
        temp_word2topic = []
        for w_idx, word in enumerate(segment):
            init_topic = random.randint(0, topic_num-1)
            file2topic[f_idx, init_topic] += 1
            topic2word[init_topic, word] += 1
            topic_count[init_topic] += 1
            temp_word2topic.append(init_topic)
        word2topic.append(temp_word2topic)
```

经过初始化之后，每个词都随机分到了一个主题，为了避免出现某一词出现极少等现象导致的除零异常，设置了$\alpha$和$beta$参数。

```python
def gibbs_sample():
    global file2topic
    global topic2word
    global topic_count
    new_file2topic = np.zeros([file_num, topic_num])
    new_topic2word = np.zeros([topic_num, word_num])
    new_topic_count = np.zeros([topic_num])
    for f_idx, segment in enumerate(segments):
        for w_idx, word in enumerate(segment):
            old_topic = word2topic[f_idx][w_idx]
            p = np.divide(np.multiply(file2topic[f_idx, :], topic2word[:, word]), topic_count)
            new_topic = np.random.multinomial(1, p/p.sum()).argmax()
            word2topic[f_idx][w_idx] = new_topic
            new_file2topic[f_idx, new_topic] += 1
            new_topic2word[new_topic, word] += 1
            new_topic_count[new_topic] += 1
    file2topic = new_file2topic
    topic2word = new_topic2word
    topic_count = new_topic_count
```

最后则迭代运行Gibbs Sampling。

### 结论

LDA模型能够较好地解决一词多义和多词一意的问题，实验说明了LDA的有效性，并通过SVM进行了验证，针对金庸小说，有效的分类通常是一些有密切关系的人物，或者有关联的动作等。

### 参考文档

[文本主题模型 LDA ](https://zhuanlan.zhihu.com/p/176929693)

[LDA主题模型的原理和建模](https://blog.csdn.net/turkeym4/article/details/113697180)

