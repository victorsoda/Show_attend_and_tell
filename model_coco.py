# -*- coding: utf-8 -*-
import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cPickle
from conf import *
import tensorflow.python.platform
from keras.preprocessing import sequence
from collections import Counter
from crop import *
import time
import datetime


class Caption_Generator():
    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def __init__(self, dim_image_L, dim_image_D, dim_embed, dim_hidden, batch_size, n_lstm_steps, n_words,
                 hard_attetion=True, selector=True, is_train=True):

        self.dim_image_L = np.int(dim_image_L)
        self.dim_image_D = np.int(dim_image_D)
        self.dim_embed = np.int(dim_embed)
        self.dim_hidden = np.int(dim_hidden)
        self.batch_size = np.int(batch_size)
        self.n_lstm_steps = np.int(n_lstm_steps)
        self.n_words = np.int(n_words)

        self.selector = selector # 是否采用注意力增强
        self.hard_attetion = hard_attetion # 是否采用hard_attetion
        self.is_train = is_train # 目前是否在训练

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # 词特征矩阵 [词表大小，特征长度]
        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_embed], -0.1, 0.1), name='Wemb')
            # 词特征偏置
            self.bemb = self.init_bias(dim_embed, name='bemb')
        # 建立LSTM CELL
        self.lstm = tf.contrib.rnn.BasicLSTMCell(num_units=dim_hidden)

    def build_model(self):
        """
        构建模型
        :return:
        """
        # N * dim_image_L(位置数量) * dim_image_D （每个位置的维度）
        image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image_L * self.dim_image_D])
        sentence = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps])
        mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])

        # 编码图像特征 ( batch_size, dim_image_L, dim_image_D)
        image_emb = tf.reshape(image, [self.batch_size, self.dim_image_L, self.dim_image_D])
        # 图像特征重构，参考github上的实现所引入 ( batch_size, dim_image_L, dim_image_D)
        image_pro = self._project_features(image_emb)

        # 获取LSTM的初始c h
        c, h = self._get_initial_lstm(features=image_emb)
        # 每一个词对应的图像位置概率分布
        alpha_list = []

        loss = 0.0
        with tf.variable_scope("RNN"):
            for i in range(self.n_lstm_steps):  # maxlen + 1

                # 获取注意力处理后的图像特征(N x D)
                image_context, alpha = self._attention_layer(features=image_emb, features_proj=image_pro, h=h,
                                                             reuse=(i != 0))
                alpha_list.append(alpha)
                if self.selector:
                    image_context, beta = self._selector(context=image_context, h=h, reuse=(i != 0))

                if i == 0:
                    # 开始状态，用开始词#START#的特征向量作为初始输入
                    temp = tf.zeros([self.batch_size, 1], tf.int32)
                    current_emb = tf.nn.embedding_lookup(self.Wemb, temp[:, 0]) + self.bemb
                else:
                    with tf.device("/cpu:0"):
                        # 通过wordindex向词特征Wemb查询词特征向量
                        current_emb = tf.nn.embedding_lookup(self.Wemb, sentence[:, i - 1]) + self.bemb

                with tf.variable_scope('lstm', reuse=(i != 0)):
                    # 将注意力和词特征级联输入LSTM
                    output, (c, h) = self.lstm(inputs=tf.concat([current_emb, image_context], 1),
                                               state=[c, h])
                # 利用隐藏层输出h，获取当前时刻的词索引
                logit_words = self._decode_lstm(h, dropout=True, reuse=(i != 0))

                if i > 0:  # 计算loss
                    # 将label的词索引转为one_hot编码
                    labels = tf.expand_dims(sentence[:, i], 1)  # (batch_size)
                    indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                    concated = tf.concat([indices, labels], 1)
                    onehot_labels = tf.sparse_to_dense(
                        concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)  # (batch_size, n_words)
                    # 计算cross_loss
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
                    # 根据每个实际句子的长度截取loss
                    cross_entropy = cross_entropy * mask[:, i]

                    current_loss = tf.reduce_sum(cross_entropy)
                    loss = loss + current_loss

            loss = loss / tf.reduce_sum(mask[:, 1:])
            return loss, image, sentence, mask

    def _project_features(self, features):
        """
        将原始图像特征作一次线性变换
        :param features: 原始图像特征
        :return: 经过简单变换的原始图像特征
        """
        with tf.variable_scope('project_features'):
            w = tf.Variable(tf.truncated_normal([self.dim_image_D, self.dim_image_D],
                                                stddev=1.0 / math.sqrt(float(self.dim_image_D))), name='w')
            features_flat = tf.reshape(features, [-1, self.dim_image_D])  # shape[batch * l, D]
            features_proj = tf.matmul(features_flat, w)  # shape[batch * l, D]
            # 整型回shape[N,L,D]
            features_proj = tf.reshape(features_proj, [-1, self.dim_image_L, self.dim_image_D])  # shape[N,L,D]
            return features_proj

    def _get_initial_lstm(self, features):
        """
        利用图像特征平均值生成lstm的初态
        :param features:
        :return: c,h
        """
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.dim_image_D, self.dim_hidden], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.dim_hidden], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.dim_image_D, self.dim_hidden], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.dim_hidden], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _selector(self, context, h, reuse=False):
        """
        增强注意力效果
        :param context: 经过概率加权的注意力特征
        :param h: 上一时刻的隐藏层输出
        :param reuse:
        :return: 经过增强的注意力特征
        """
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.dim_hidden, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')  # (N, 1)
            context = tf.multiply(beta, context, name='selected_context')
            return context, beta

    def _attention_layer(self, features, features_proj, h, reuse=False):
        """
        注意力特征生成
        :param features:原始图像特征
        :param features_proj: 经过简单变化的图像特征
        :param h: 上一时刻隐藏层输入
        :param reuse:
        :return:
        """
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.dim_hidden, self.dim_image_D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.dim_image_D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.dim_image_D, 1], initializer=self.weight_initializer)

            w_f = tf.get_variable('w_f', [self.dim_hidden, 1], initializer=self.weight_initializer)
            w_h = tf.get_variable('w_h', [self.dim_hidden, 1], initializer=self.weight_initializer)

            # 通道特征选择系数
            fatt_l = tf.expand_dims(tf.matmul(h, w_f), 1)
            fatt_d = tf.expand_dims(tf.matmul(h, w_h), 1)
            bia = tf.expand_dims(tf.matmul(h, w), 1)

            # 计算通道选择开关
            h_att = tf.nn.tanh(features_proj * fatt_l + bia * fatt_d + b)  # (N, L, D)
            # 计算各个图像位置的比重
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.dim_image_D]), w_att),
                                 [-1, self.dim_image_L])  # (N, L)
            # 计算各个位置softmax概率
            alpha = tf.nn.softmax(out_att)  # N*L

            if self.hard_attetion:
                # hard-attention
                max_index = tf.cast(tf.expand_dims(tf.argmax(alpha, 1), 1), dtype=tf.int32)  # N*1
                index = tf.expand_dims(tf.range(0, self.batch_size), 1)
                concated = tf.concat([index, max_index], 1)
                out_shape = tf.pack([self.batch_size, self.dim_image_L])
                alpha_hard = tf.sparse_to_dense(concated, out_shape,1.0, 0.0)
                # 图像特征的加权求和,alpha_hard为[0,0,0,....,1,...,0,0,0,0]的形式
                context = tf.reduce_sum(features * tf.expand_dims(alpha_hard, 2), 1, name='context')  # (N, D)
            else:
                # soft-attention
                context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')  # (N, D)

            return context, alpha

    def _decode_lstm(self, h, dropout=False, reuse=False):
        """
        解码，通过隐藏层输出获取当前时刻的词索引
        :param h:
        :param dropout:
        :param reuse:
        :return:
        """
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.dim_hidden, self.dim_embed], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.dim_embed], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.dim_embed, self.n_words], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.n_words], initializer=self.const_initializer)
            # 预测词特征向量
            h_logits = tf.matmul(h, w_h) + b_h
            h_logits = tf.nn.relu(h_logits)
            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            # 预测词索引的概率分布
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    def build_generator(self, maxlen):
        """
        句子生成器
        :param maxlen:
        :return:
        """
        # N * dim_image_L(位置数量) * dim_image_D （每个位置的维度）
        image = tf.placeholder(tf.float32, [1, self.dim_image_L, self.dim_image_D])
        # 编码图像特征 ( 1, dim_image_L, dim_image_D)
        image_emb = image
        # 图像特征重构，参考github上的实现所引入 ( 1, dim_image_L, dim_image_D)
        image_pro = self._project_features(image_emb)

        # 获取LSTM的初始c h
        c, h = self._get_initial_lstm(features=image_emb)
        # 每一个词对应的图像位置概率分布
        alpha_list = []
        # 每一个词对应的注意力增强系数
        beta_list = []
        # 每一个词的索引
        generated_words = []
        with tf.variable_scope("RNN"):
            # 初始状态，词特征为#START#的词特征
            first_word = tf.nn.embedding_lookup(self.Wemb, [0]) + self.bemb
            for i in range(maxlen):
                image_context, alpha = self._attention_layer(image_emb, image_pro, h, reuse=(i != 0))
                alpha_list.append(alpha)

                if self.selector:
                    image_context, beta = self._selector(image_context, h, reuse=(i != 0))
                    beta_list.append(beta)

                if i == 0:
                    current_emb = first_word

                with tf.variable_scope('lstm', reuse=(i != 0)):
                    output, (c, h) = self.lstm(inputs=tf.concat([current_emb, image_context], 1), state=[c, h])

                # 解码词索引概率分布
                logit_words = self._decode_lstm(h, dropout=False, reuse=(i != 0))
                # 词索引概率最大的词为该时刻生成的词
                max_prob_word = tf.argmax(logit_words, 1)

                with tf.device("/cpu:0"):
                    # 当前时刻生成的词特征同时作为下一时刻的词特征输入
                    current_emb = tf.nn.embedding_lookup(self.Wemb, max_prob_word) + self.bemb

                if i > 0:
                    generated_words.append(max_prob_word)

        return image, generated_words, alpha_list


def get_caption_data(annotation_path, feat_path):
    feats = np.load(feat_path)
    annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['caption'])
    captions = annotations['caption'].values

    return feats, captions


def preProBuildWordVocab(sentence_iterator, word_count_threshold=30):  # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold,)
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0  # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0 * word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector)  # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)  # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector





def train():
    learning_rate = 0.0001
    momentum = 0.0
    training_size = 80000
    
    feats, captions = get_caption_data(annotation_path, feat_path)
    feats = feats[:training_size]
    captions = captions[:training_size]
    
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions)
    tmp = feats.shape
    feats = feats.reshape((tmp[0], tmp[2]))

    np.save('data/ixtoword_COCO', ixtoword)

    index = np.arange(len(feats))
    np.random.shuffle(index)

    feats = feats[index]
    captions = captions[index]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    n_words = len(wordtoix)
    max_len = np.max(map(lambda x: len(x.split(' ')), captions))

    caption_generator = Caption_Generator(
        dim_image_L=dim_image_L,
        dim_image_D=dim_image_D,
        dim_hidden=dim_hidden,
        dim_embed=dim_embed,
        batch_size=batch_size,
        n_lstm_steps=max_len + 2,
        n_words=n_words,
        hard_attetion=False
        )

    loss, image, sentence, mask = caption_generator.build_model()
    # 绘图
    tf.summary.scalar('batch_loss', loss)

    summary_op = tf.summary.merge_all()

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # 初始化图中所有变量参数
    tf.global_variables_initializer().run()
    summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

    # 加载之前的训练模型
    saver = tf.train.Saver(max_to_keep=50)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print 'loading......'
        saver.restore(sess, ckpt.model_checkpoint_path)
    print "image_feature shape: ", len(feats), "\n"

    for epoch in range(1, n_epochs):
        start_t = time.time()
        # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        for start, end in zip(
                range(0, len(feats), batch_size),
                range(batch_size, len(feats), batch_size)
        ):
            # 获取该batch的图像特征(NxLxD)
            current_feats = feats[start:end]
            # 获取该batch的captions
            current_captions = captions[start:end]
            # 将每个句子转成词的索引向量(N x len) len为句子长度 N为这个batch的大小
            current_caption_ind = map(
                lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix],
                current_captions)
            # 将所有句子补0成相同长度
            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=max_len + 1)
            # 将句子索引向量列表转换成 N × max_len长的索引矩阵，每一行为一个句子的索引列表
            current_caption_matrix = np.hstack(
                [np.full((len(current_caption_matrix), 1), 0), current_caption_matrix]).astype(int)

            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array(map(lambda x: (x != 0).sum() + 2, current_caption_matrix))
            #  +2 -> #START# and '.'

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            feed_dict = {
                image: current_feats,
                sentence: current_caption_matrix,
                mask: current_mask_matrix
            }

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            summary = sess.run(summary_op, feed_dict)
            summary_writer.add_summary(summary, end / batch_size)

            rate = float(start) * 100 / len(feats)
            rate = round(rate, 2)
            print str(datetime.datetime.now())[:-7], "Epoch", epoch, ",", '%.2f' % rate, '%' + ' complete.', "Current Cost:", loss_value

        print "Epoch ", epoch, " is done." + " Used time " + str(time.time() - start_t) + " Saving the model ... "
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
        learning_rate *= 0.96


def read_image(path):
    img = crop_image(path, target_height=224, target_width=224)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    img = img[None, ...]
    return img


