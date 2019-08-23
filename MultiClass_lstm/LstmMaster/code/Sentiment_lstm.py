import multiprocessing

import os
import pandas as pd
import numpy as np
import jieba
import yaml
from gensim.corpora import Dictionary
from gensim.models import Word2Vec
from keras import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, Activation
from keras.models import model_from_yaml
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

vocab_dim = 100
n_exposures = 10
window_size = 7
cpu_count = multiprocessing.cpu_count()
n_iterations = 1
max_len = 10
input_length = 100
batch_size = 32
n_epoch = 4
   # 这是二分类的文件

# 加载文件
def loadfile():
    neg = pd.read_excel('../data/neg.xlsx', sheet_name=0, header=None, index=None)
    pos = pd.read_excel('../data/pos.xls', sheet_name=0, header=None, index=None)
    # combined 就是全部的句子都整合到一起了。
    combined = np.concatenate((pos[0], neg[1]))
    y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)))
    print(combined,'-----------------------')
    return combined, y

#loadfile()

# 对句子进行分词，并取掉换行符
def tokenizer(text):
    '''Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引  设置向量为100 维，窗口大小为7  忽略掉词频低于5的词。 worker 制定了cpu线程数目。
def word2vec_train(combined):
    #建立 空的模型对象
    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    #遍历一次语料库建立词典
    model.build_vocab(combined)
    #遍历语料库建立神经网络模型
    model.train(combined, total_examples=model.corpus_count, epochs=model.iter)
    # 保存模型。
    model.save('../lstm_data/Word2vec_model.pkl')
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        combined=None):
    maxlen = 100
    ''' Function does are number of Jobs:
            1- Creates a word to index mapping
            2- Creates a word to vector mapping
            3- Transforms the Training and Testing Dictionaries
    '''
    if (combined is not None) and model is not None:
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(), allow_update=True)
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有词频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}  # 所有词频数超过10的词语的词向量

        #print(w2vec)
        def parse_dataset(combined):
            '''
            Words become integers
            '''
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)
        return w2indx, w2vec, combined
    else:
        print('No data provide')

# 返回真正模型要用的data。
def get_data(index_dict, word_vectors, combined, y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，词频小于10的词语索引为0，所以加1

    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 索引为0的词语，词向量全为0

    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    print(x_train, x_test, y_train, y_test)
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


# 定义网络结构
def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    print('Defining a simple Keras Model')
    model = Sequential()  # or Graph or whatever  output_dim 词向量的维度 input_dim：词汇表的大小 input_length :输入序列的长度。
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))

    model.add(LSTM(activation="sigmoid", units=50, recurrent_activation="hard_sigmoid"))
    #rate: 在 0 和 1 之间浮动。需要丢弃的输入比例
    model.add(Dropout(0.5))
    # 指定输出尺寸。
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print('Compiling the Model...')

    #损失函数，优化器，训练和测试期间的模型评估标准
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print("Train...")
    # 以给定数量的轮次 训练模型。
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch, verbose=1, validation_data=(x_test, y_test))

    print("Evaluate...")
    #训练和测试期间的模型评估标准  返回模型的误差值和评估标准值。
    score = model.evaluate(x_test, y_test,
                           batch_size=batch_size)
    #YAML 字符串的形式返回模型的表示
    yaml_string = model.to_yaml()
    with open('../lstm_data/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    #保留权重文件。
    model.save_weights('../lstm_data/lstm.h5')
    print('Test score:', score)


# 把上面的函数连接起来使用。 完成模型的训练
def train():
    print('Loading Data...')
    combined, y = loadfile()
    print(len(combined), len(y))
    print('Tokenising...')
    combined = tokenizer(combined)
    print('Training a Word2vec model...')
    index_dict, word_vectors, combined = word2vec_train(combined)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
    print(x_train.shape, y_train.shape)
    train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)


def input_transform(string):
    words = jieba.lcut(string)
    words = np.array(words).reshape(1, -1)
    model = Word2Vec.load('MultiClass_lstm/LstmMaster/lstm_data/Word2vec_model.pkl')
    _, _, combined = create_dictionaries(model, words)
    return combined


# 执行结果
def lstm_predict(string):
    print('loading model......')
    print(os.path.realpath(__file__))
    with open('MultiClass_lstm/LstmMaster/lstm_data/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('MultiClass_lstm/LstmMaster/lstm_data/lstm.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    data = input_transform(string)
    data.reshape(1, -1)


    result = model.predict_classes(data)
    string = jieba.lcut(string)
    print(result)
    if result[0][0] == 1:
        print(string, ' \n积极')
        return "积极"
    else:
        print(string, ' \n消极')
        return "消极"


if __name__ == '__main__':
    #train()
    #string = '作者在战几时之前用了＂拥抱＂令人叫绝。日本如果没有战败，就有会有美军的占领，没胡官僚主义的延续，没有战后的民发反思，没有，就不会让日本成为一个经济强国。当然，美国人也给日本人带来了耻辱。对日中关系也造成了深远的影响。文中揭露了“东京审判”中很多鲜为人知的东西。让人惊醒。唉！中国人民对日本的了解是不是太少了。'
    #string = '做为一本声名在外的流行书，说的还是广州的外企，按道理应该和我的生存环境差不多啊。但是一看之下，才发现相去甚远。这也就算了，还发现其中的很多规则有很强的企业个性，也就说，只是个例，而不是行例。给我们这些老油条看看也就算了，如果给那些对外企向往，或者想了解的freshman来看，实在是容易误导他们。'
    string = '你们这个酒店的价格真的好贵啊，真的性价比低。。。'
    lstm_predict(string)
