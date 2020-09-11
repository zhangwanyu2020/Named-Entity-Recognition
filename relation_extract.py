import numpy as np
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense
from tools import f1_np, focal_loss, get_weight
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # '-1' means use cpu

num_classes = 44
maxlen = 350
batch_size = 16
config_path = '/content/drive/My Drive/extract/albert_config.json'
checkpoint_path = '/content/drive/My Drive/extract/model.ckpt'
dict_path = '/content/drive/My Drive/extract/vocab_chinese.txt'


def load_data(filepath):
    lines = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            l_split = line.split('fengefu')
            text = l_split[0]
            labels = l_split[1]
            lines.append((text, labels))
    return lines


datas = load_data('/content/drive/My Drive/extract/relation_train.csv')
# print(len(datas)) # 14339
train_data = datas[:int(len(datas) * 0.9)]
print(len(train_data))  # 11471
valid_data = datas[int(len(datas) * 0.9):]
print(len(valid_data))  # 2868

tokenizer = Tokenizer(dict_path, do_lower_case=True)


class dataGenerator(DataGenerator):
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            label = label.split(' ')
            label_ = [0] * num_classes
            for k in label:
                label_[int(k)] = 1
            batch_labels.append(label_)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='albert',
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
output = Dense(
    units=num_classes,
    activation='sigmoid',
    kernel_initializer=bert.initializer
)(output)
model = keras.models.Model(bert.model.input, output)
model.summary()

AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
model.compile(
    loss=focal_loss(gamma=1, alpha=0.9),  # get_weight(weight_1=80,weight_0=20), 'binary_crossentropy'
    optimizer=Adam(1e-5),  # 用足够小的学习率
    metrics=['accuracy']
)

train_generator = dataGenerator(train_data, batch_size)


def evaluate(data):
    mi = []
    ma = []
    for (text, labels) in data:
        labels = labels.split(' ')
        label_true = [0] * num_classes
        for k in labels:
            label_true[int(k)] = 1
        token_ids, segment_ids = tokenizer.encode(text)
        res = model.predict([[token_ids], [segment_ids]])[0]
        res1 = res.reshape((len(res), 1))
        # print('res1:',res1)
        res = list(np.where(res1 > 0.5)[0])
        # print('res2:',res)
        label_pred = [0] * num_classes
        for r in res:
            label_pred[int(r)] = 1
        label_true = np.array(label_true)
        label_pred = np.array(label_pred)
        # print(label_true)
        # print(label_pred)

        micro_f1, macro_f1 = f1_np(label_true, label_pred)
        mi.append(micro_f1)
        ma.append(macro_f1)
    print('res1:', res1)
    print(label_true)
    print(label_pred)
    mi = sum(mi) / len(mi)
    ma = sum(ma) / len(ma)

    return mi, ma


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.macro_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        micro_f1, macro_f1 = evaluate(valid_data)
        if macro_f1 > self.macro_f1:
            self.macro_f1 = macro_f1
            model.save_weights('/content/sample_data/best_model.weights')
        print(
            u'macro f1: %.5f, best macro f1: %.5f\n' %
            (macro_f1, self.macro_f1)
        )


# 输入文本做测试
def predict_re(text):
    model.load_weights('/content/sample_data/best_model.weights')
    token_ids, segment_ids = tokenizer.encode(text)
    res = model.predict([[token_ids], [segment_ids]])[0]
    res1 = res.reshape((len(res), 1))
    res2 = list(np.where(res1 > 0.5)[0])
    return res1, res2


# 用验证集做测试
def predict_re2(valid_data):
    model.load_weights('/content/sample_data/best_model.weights')
    pred_index = []
    for (text, labels) in valid_data:
        labels = labels.split(' ')
        label_true = [0] * num_classes
        for k in labels:
            label_true[int(k)] = 1
        token_ids, segment_ids = tokenizer.encode(text)
        res = model.predict([[token_ids], [segment_ids]])[0]
        res1 = res.reshape((len(res), 1))
        # print('res1:',res1)
        res2 = list(np.where(res1 > 0.5)[0])
        # print('res2:',res)
        label_pred = [0] * num_classes
        for r in res2:
            label_pred[int(r)] = 1
        label_true = np.array(label_true)
        label_pred = np.array(label_pred)
        pred_index.append(res2)
    return pred_index, res1, res2, label_true


if __name__ == "__main__":
    evaluator = Evaluator()
    # model.load_weights('/content/sample_data/best_model2.weights') 重载权重
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=25,
        callbacks=[evaluator]
    )

    # #test
    # pred_index,res1,res2,true_label = predict_re2(valid_data)
    # print(pred_index,res1,res2,true_label)

