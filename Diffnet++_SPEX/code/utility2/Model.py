import time

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers, initializers
import math
from utility2.GAL import GraphAttentionLayer

class TRUST_PATH(tf.keras.Model):
    def __init__(self, opt):
        super(TRUST_PATH, self).__init__()
        # trust
        self.opt = opt
        self.initializeNodes()

    def initializeNodes(self):
        stdv = 1.0 / math.sqrt(self.opt.hidden_size)
        self.user_embedding = tf.Variable(tf.random.normal([self.opt.num_users + 1, self.opt.hidden_size], stddev=0.01),name='user_embedding')

        # self.user_embedding = layers.Embedding(input_dim=self.opt.num_users + 1,
        #                                        output_dim=self.opt.hidden_size,
        #                                        embeddings_initializer=initializers.RandomNormal(mean=0., stddev=1.),
        #                                        name='user_embedding')
        self.linear_one = layers.Dense(units=self.opt.hidden_size, use_bias=True,
                                       kernel_initializer=tf.random_uniform_initializer(-stdv, stdv))
        self.linear_two = layers.Dense(units=self.opt.hidden_size, use_bias=True,
                                       kernel_initializer=tf.random_uniform_initializer(-stdv, stdv))
        self.linear_three = layers.Dense(units=1, use_bias=False,
                                         kernel_initializer=tf.random_uniform_initializer(-stdv, stdv))
        self.linear_transform = layers.Dense(units=self.opt.hidden_size, use_bias=True,
                                             kernel_initializer=tf.random_uniform_initializer(-stdv, stdv))
        # multi head
        self.in_att = [GraphAttentionLayer(self.opt.hidden_size, concat=True) for _ in range(self.opt.nb_heads)]
        self.out_att = GraphAttentionLayer(self.opt.hidden_size, concat=False)

        initializer = tf.initializers.GlorotUniform()
        self.w = tf.Variable(initializer(shape=[self.opt.nb_heads * self.opt.hidden_size, self.opt.hidden_size]), name="TRUST_PATH_w")


    def call(self, dataset, slice_indices, flag=True):
        path, mask, targets = dataset.get_slice(slice_indices)
        # multi head
        seq_l = tf.reduce_sum(mask, 1)  # path length
        mul_seq = tf.concat([att(self.user_embedding, path, seq_l) for att in self.in_att], 2)
        mul_seq_c = tf.concat([mul_seq[i] for i in range(mul_seq.get_shape()[0])], 0)
        mul_one = tf.matmul(mul_seq_c, self.w)
        mul_one = tf.nn.elu(mul_one)
        seq_hidden = self.out_att(self.user_embedding, tf.reshape(mul_one,
                                                                  [mul_seq.get_shape()[0], mul_seq.get_shape()[1],
                                                                   self.opt.hidden_size]), seq_l)

        last_ind = tf.stack([tf.range(mask.get_shape()[0]), tf.cast(seq_l - 1, dtype=tf.int32)], 1)
        ht = tf.gather_nd(seq_hidden, last_ind)  # batch_size x latent_size
        q1 = tf.expand_dims(self.linear_one(ht), 1)  # batch_size x 1 x latent_size
        q2 = self.linear_two(seq_hidden)             # batch_size x seq_length x latent_size
        alpha = self.linear_three(tf.sigmoid(q1 + q2))
        a = tf.reduce_sum(alpha * seq_hidden * tf.reshape(tf.cast(mask, dtype=tf.float32), [mask.get_shape()[0], -1, 1]),1)  # float
        if not self.opt.nonhybrid:
            a = self.linear_transform(tf.concat([a, ht], 1))
        b = self.user_embedding[:-1]  # n_nodes x latent_size
        scores = tf.matmul(a, tf.transpose(b, perm=[1, 0]))

        if flag:
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.keras.utils.to_categorical(targets,
                                                           num_classes=self.opt.num_users),
                                                           logits=scores)
            loss = tf.reduce_mean(loss)
            return loss
        else:
            return scores, targets
