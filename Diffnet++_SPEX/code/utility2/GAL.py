import tensorflow as tf

class GraphAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, hidden_size, concat=True):
        super().__init__()
        self.concat = concat
        self.hidden_size = hidden_size

        initializer = tf.initializers.GlorotUniform()
        self.a = tf.Variable(initializer(shape=[2 * hidden_size, 1]), name="GAL_a")


    # @tf.function
    # def process(self, emb, seq_l_p, seq_p_i, seq_p_i_1):
    #
    #     ue_pos_emb = tf.cast(tf.repeat(seq_l_p, repeats=self.hidden_size), tf.float32)
    #     fe_pos_emb = tf.cast(tf.repeat(seq_l_p - 1, repeats=self.hidden_size), tf.float32)
    #     u = emb[seq_p_i] + ue_pos_emb
    #     ue = tf.stack([u, u], 0)
    #     fe = tf.stack([u, emb[seq_p_i_1] + fe_pos_emb], 0)  # [2, 64]
    #     h = tf.concat([ue, fe], 1)  # [2, 128]
    #     att = tf.matmul(h, self.a)  # [2, 128] * [128, 1] = [2, 1]
    #     att = tf.nn.softmax(att, 0)  # [2, 1]
    #     u = tf.squeeze(tf.matmul(tf.transpose(att, perm=[1, 0]), fe), 0)  # [1,2] * [2,64] = [1,64] -> [64]
    #
    #     return u

    # @tf.function
    # def process2(self, seq_p_i, seq_p_i_1):
    #
    #     ue = tf.stack([seq_p_i, seq_p_i], 0)                # [2, 64]
    #     fe = tf.stack([seq_p_i, seq_p_i_1], 0)
    #     h = tf.concat([ue, fe], 1)                          # [2, 128]
    #     att = tf.matmul(h, self.a)                          # [2, 128] * [128, 1] = [2, 1]
    #     att = tf.nn.softmax(att, 0)                         # [2, 1]
    #     u = tf.squeeze(tf.matmul(tf.transpose(att, perm=[1, 0]), fe), 0)  # [1, 2] * [2, 64]  = [1, 64]
    #
    #     return u


    def call(self, emb, seq, seq_l):

        global_results = []  # [135, 5, 64]

        if self.concat:
            for p in range(seq.get_shape()[0]):
                results = []

                seq_l_p = seq_l[p]
                seq_p = seq[p]
                # print("seq_l_p:",seq_l_p,list(range(seq_l_p - 1)),list(range(seq_l_p - 1, seq.get_shape()[1])))

                for i in range(seq_l_p - 1):
                    seq_p_i = seq_p[i]
                    seq_p_i_1 = seq_p[i+1]
                    # results.append(self.process(emb, seq_l_p, seq_p_i, seq_p_i_1))
                    ue_pos_emb = tf.cast(tf.repeat(seq_l_p, repeats=self.hidden_size), tf.float32)
                    fe_pos_emb = tf.cast(tf.repeat(seq_l_p - 1, repeats=self.hidden_size), tf.float32)
                    u = emb[seq_p_i] + ue_pos_emb
                    ue = tf.stack([u, u], 0)
                    fe = tf.stack([u, emb[seq_p_i_1] + fe_pos_emb], 0)  # [2, 64]
                    h = tf.concat([ue, fe], 1)  # [2, 128]
                    att = tf.matmul(h, self.a)  # [2, 128] * [128, 1] = [2, 1]
                    att = tf.nn.softmax(att, 0)  # [2, 1]
                    u = tf.squeeze(tf.matmul(tf.transpose(att, perm=[1, 0]), fe), 0)  # [1,2] * [2,64] = [1,64] -> [64]
                    results.append(u)


                for i in range(seq_l_p - 1, seq.get_shape()[1]):
                    results.append(emb[seq_p[i]])
                global_results.append(tf.stack(results, 0))

            return tf.stack(global_results, 0)
        else:
            for p in range(tf.shape(seq)[0]):
                results = []
                seq_l_p = seq_l[p]
                seq_p = seq[p]

                for i in range(seq_l_p - 1):
                    seq_p_i = seq_p[i]
                    seq_p_i_1 = seq_p[i + 1]
                    # results.append(self.process2(seq_p_i, seq_p_i_1))
                    ue = tf.stack([seq_p_i, seq_p_i], 0)  # [2, 64]
                    fe = tf.stack([seq_p_i, seq_p_i_1], 0)
                    h = tf.concat([ue, fe], 1)  # [2, 128]
                    att = tf.matmul(h, self.a)  # [2, 128] * [128, 1] = [2, 1]
                    att = tf.nn.softmax(att, 0)  # [2, 1]
                    u = tf.squeeze(tf.matmul(tf.transpose(att, perm=[1, 0]), fe), 0)  # [1, 2] * [2, 64]  = [1, 64]
                    results.append(u)

                for i in range(seq_l_p - 1, seq.get_shape()[1]):
                    results.append(seq_p[i])
                global_results.append(tf.stack(results, 0))
            return tf.stack(global_results, 0)
