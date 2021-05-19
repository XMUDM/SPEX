from __future__ import division
import tensorflow as tf
from tensorflow.keras import layers
from utility2.GAL import GraphAttentionLayer

class MultiTaskModel(tf.keras.Model):
    def __init__(self, conf, auto=True):
        super(MultiTaskModel, self).__init__()
        self.conf = conf
        self.supply_set = (
            'SOCIAL_NEIGHBORS_SPARSE_MATRIX',
            'CONSUMED_ITEMS_SPARSE_MATRIX',
            'ITEM_CUSTOMER_SPARSE_MATRIX'
        )
        self.user_embedding = tf.Variable(tf.random.normal([self.conf.num_users+1, self.conf.hidden_size], stddev=0.01),name='user_embedding')
        self.item_embedding = tf.Variable(tf.random.normal([self.conf.num_items, self.conf.hidden_size], stddev=0.01),name='item_embedding')
        if auto:
            self.task_weights = tf.Variable([0.5, 0.5], name='task_weights')


        # expert_s------------------------------------
        initializer1 = tf.initializers.GlorotUniform()
        initializer2 = tf.initializers.GlorotUniform()
        self.att_exp1 = tf.Variable(initializer1(shape=[3 * self.conf.hidden_size, 3]), name="att1")
        self.att_exp2 = tf.Variable(initializer2(shape=[3 * self.conf.hidden_size, 3]), name="att2")
        # -----------------------------------------

    def convertDistribution(self, x):
        mean, var = tf.nn.moments(x=x, axes=[0, 1])
        y = (x - mean) * 0.1 / tf.sqrt(var)
        return y

    # ----------------------
    # Operations for Diffusion
    def generateUserEmbeddingFromSocialNeighbors(self, current_user_embedding):
        user_embedding_from_social_neighbors = tf.sparse.sparse_dense_matmul(
            self.social_neighbors_sparse_matrix_avg, current_user_embedding
        )
        return user_embedding_from_social_neighbors

    def generateUserEmbeddingFromSocialNeighbors1(self, current_user_embedding):
        user_embedding_from_social_neighbors = tf.sparse.sparse_dense_matmul(
            self.first_social_neighbors_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_social_neighbors

    def generateUserEmebddingFromConsumedItems(self, current_item_embedding):
        user_embedding_from_consumed_items = tf.sparse.sparse_dense_matmul(
            self.consumed_items_sparse_matrix_avg, current_item_embedding
        )
        return user_embedding_from_consumed_items

    def generateUserEmebddingFromConsumedItems1(self, current_item_embedding):
        user_embedding_from_consumed_items = tf.sparse.sparse_dense_matmul(
            self.first_consumed_items_low_level_att_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items

    def generateItemEmebddingFromCustomer(self, current_user_embedding):
        item_embedding_from_customer = tf.sparse.sparse_dense_matmul(
            self.item_customer_sparse_matrix_avg, current_user_embedding
        )
        return item_embedding_from_customer

    def generateItemEmebddingFromCustomer1(self, current_user_embedding):
        item_embedding_from_customer = tf.sparse.sparse_dense_matmul(
            self.first_items_users_neighborslow_level_att_matrix, current_user_embedding
        )
        return item_embedding_from_customer

    def generateUserEmbeddingFromSocialNeighbors2(self, current_user_embedding):
        user_embedding_from_social_neighbors = tf.sparse.sparse_dense_matmul(
            self.second_social_neighbors_low_level_att_matrix, current_user_embedding
        )
        return user_embedding_from_social_neighbors

    def generateUserEmebddingFromConsumedItems2(self, current_item_embedding):
        user_embedding_from_consumed_items = tf.sparse.sparse_dense_matmul(
            self.second_consumed_items_low_level_att_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items

    def generateItemEmebddingFromCustomer2(self, current_user_embedding):
        item_embedding_from_customer = tf.sparse.sparse_dense_matmul(
            self.second_items_users_neighborslow_level_att_matrix, current_user_embedding
        )
        return item_embedding_from_customer


    def generateUserEmebddingFromConsumedItems3(self, current_item_embedding):
        user_embedding_from_consumed_items = tf.sparse.sparse_dense_matmul(
            self.third_consumed_items_low_level_att_matrix, current_item_embedding
        )
        return user_embedding_from_consumed_items

    def generateItemEmebddingFromCustomer3(self, current_user_embedding):
        item_embedding_from_customer = tf.sparse.sparse_dense_matmul(
            self.third_items_users_neighborslow_level_att_matrix, current_user_embedding
        )
        return item_embedding_from_customer

    def initializeNodes(self, data_dict):

        self.reduce_dimension_layer = tf.keras.layers.Dense(units=self.conf.hidden_size, activation=tf.nn.sigmoid,
                                                            name='reduce_dimension_layer')

        ########  Fine-grained Graph Attention initialization ########

        # ----------------------
        # First diffusion layer
        self.first_user_part_social_graph_att_layer1 = tf.keras.layers.Dense(units=1, activation=tf.nn.tanh,
                                                                             name='firstGCN_UU_user_MLP_first_layer')
        self.first_user_part_social_graph_att_layer2 = tf.keras.layers.Dense(units=1, activation=tf.nn.leaky_relu,
                                                                             name='firstGCN_UU_user_MLP_sencond_layer')
        self.first_user_part_interest_graph_att_layer1 = tf.keras.layers.Dense(units=1, activation=tf.nn.tanh,
                                                                               name='firstGCN_UI_user_MLP_first_layer')
        self.first_user_part_interest_graph_att_layer2 = tf.keras.layers.Dense(units=1, activation=tf.nn.leaky_relu,
                                                                               name='firstGCN_UI_user_MLP_second_layer')

        # ----------------------
        # Second diffusion layer
        self.second_user_part_social_graph_att_layer1 = tf.keras.layers.Dense( \
            units=1, activation=tf.nn.tanh, name='secondGCN_UU_user_MLP_first_layer')

        self.second_user_part_social_graph_att_layer2 = tf.keras.layers.Dense( \
            units=1, activation=tf.nn.leaky_relu, name='secondGCN_UU_user_MLP_second_layer')

        self.second_user_part_interest_graph_att_layer1 = tf.keras.layers.Dense( \
            units=1, activation=tf.nn.tanh, name='secondGCN_UI_user_MLP_first_layer')

        self.second_user_part_interest_graph_att_layer2 = tf.keras.layers.Dense( \
            units=1, activation=tf.nn.leaky_relu, name='secondGCN_UI_user_MLP_second_layer')

        # ----------------------
        # Item part
        self.first_item_part_itself_graph_att_layer1 = tf.keras.layers.Dense( \
            units=1, activation=tf.nn.tanh, name='firstGCN_IU_itemself_MLP_first_layer')

        self.first_item_part_itself_graph_att_layer2 = tf.keras.layers.Dense( \
            units=1, activation=tf.nn.leaky_relu, name='firstGCN_IU_itemself_MLP_second_layer')

        self.first_item_part_user_graph_att_layer1 = tf.keras.layers.Dense( \
            units=1, activation=tf.nn.tanh, name='firstGCN_IU_customer_MLP_first_layer')

        self.first_item_part_user_graph_att_layer2 = tf.keras.layers.Dense( \
            units=1, activation=tf.nn.leaky_relu, name='firstGCN_IU_customer_MLP_second_layer')

        self.second_item_part_itself_graph_att_layer1 = tf.keras.layers.Dense( \
            units=1, activation=tf.nn.tanh, name='secondGCN_IU_itemself_MLP_first_layer')

        self.second_item_part_itself_graph_att_layer2 = tf.keras.layers.Dense( \
            units=1, activation=tf.nn.leaky_relu, name='secondGCN_IU_itemself_MLP_second_layer')

        self.second_item_part_user_graph_att_layer1 = tf.keras.layers.Dense( \
            units=1, activation=tf.nn.tanh, name='secondGCN_IU_customer_MLP_first_layer')

        self.second_item_part_user_graph_att_layer2 = tf.keras.layers.Dense( \
            units=1, activation=tf.nn.leaky_relu, name='secondGCN_IU_customer_MLP_second_layer')


        low_att_std = 1.0
        self.social_neighbors_indices_input = data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT']
        self.social_neighbors_values_input = data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT']
        self.user_item_sparsity_dict = data_dict['USER_ITEM_SPARSITY_DICT']
        self.consumed_items_indices_input = data_dict['CONSUMED_ITEMS_INDICES_INPUT']
        self.consumed_items_values_input = data_dict['CONSUMED_ITEMS_VALUES_INPUT']
        self.item_customer_indices_input = data_dict['ITEM_CUSTOMER_INDICES_INPUT']
        self.item_customer_values_input = data_dict['ITEM_CUSTOMER_VALUES_INPUT']

        self.first_low_att_layer_for_social_neighbors_layer1 = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid,
                                                                                     name='first_low_att_SN_layer1')
        self.first_low_att_layer_for_social_neighbors_layer2 = tf.keras.layers.Dense(units=1,
                                                                                     activation=tf.nn.leaky_relu,
                                                                                     name='first_low_att_SN_layer2')

        self.second_low_att_layer_for_social_neighbors_layer1 = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid,
                                                                                      name='second_low_att_SN_layer1')
        self.second_low_att_layer_for_social_neighbors_layer2 = tf.keras.layers.Dense(units=1,
                                                                                      activation=tf.nn.leaky_relu,
                                                                                      name='second_low_att_SN_layer2')
        self.first_low_att_layer_for_user_item_layer1 = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid,
                                                                              name='first_low_att_UI_layer1')
        self.first_low_att_layer_for_user_item_layer2 = tf.keras.layers.Dense(units=1, activation=tf.nn.leaky_relu,
                                                                              name='first_low_att_UI_layer2')
        self.second_low_att_layer_for_user_item_layer1 = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid,
                                                                               name='second_low_att_UI_layer1')
        self.second_low_att_layer_for_user_item_layer2 = tf.keras.layers.Dense(units=1, activation=tf.nn.leaky_relu,
                                                                               name='second_low_att_UI_layer2')
        self.first_low_att_layer_for_item_user_layer1 = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid,
                                                                              name='first_low_att_IU_layer1')
        self.first_low_att_layer_for_item_user_layer2 = tf.keras.layers.Dense(units=1, activation=tf.nn.leaky_relu,
                                                                              name='first_low_att_IU_layer2')
        self.second_low_att_layer_for_item_user_layer1 = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid,
                                                                               name='second_low_att_IU_layer1')
        self.second_low_att_layer_for_item_user_layer2 = tf.keras.layers.Dense(units=1, activation=tf.nn.leaky_relu,
                                                                               name='second_low_att_IU_layer2')

        self.snii1 = tf.Variable(tf.random.normal([len(self.social_neighbors_indices_input)], stddev=low_att_std))
        self.snii2 = tf.Variable(tf.random.normal([len(self.social_neighbors_indices_input)], stddev=1.0))
        self.ciii1 = tf.Variable(tf.random.normal([len(self.consumed_items_indices_input)], stddev=low_att_std))
        self.ciii2 = tf.Variable(tf.random.normal([len(self.consumed_items_indices_input)], stddev=1.0))
        self.icii1 = tf.Variable(tf.random.normal([len(self.item_customer_indices_input)], stddev=low_att_std))
        self.icii2 = tf.Variable(tf.random.normal([len(self.item_customer_indices_input)], stddev=0.01))

        # prepare the shape of sparse matrice
        self.social_neighbors_dense_shape = [self.conf.num_users, self.conf.num_users]
        self.consumed_items_dense_shape = [self.conf.num_users, self.conf.num_items]
        self.item_customer_dense_shape = [self.conf.num_items, self.conf.num_users]


    def initializeNodes2(self):
        import math
        stdv = 1.0 / math.sqrt(self.conf.hidden_size)

        self.linear_one = layers.Dense(units=self.conf.hidden_size, use_bias=True,
                                       kernel_initializer=tf.random_uniform_initializer(-stdv, stdv))
        self.linear_two = layers.Dense(units=self.conf.hidden_size, use_bias=True,
                                       kernel_initializer=tf.random_uniform_initializer(-stdv, stdv))
        self.linear_three = layers.Dense(units=1, use_bias=False,
                                         kernel_initializer=tf.random_uniform_initializer(-stdv, stdv))
        self.linear_transform = layers.Dense(units=self.conf.hidden_size, use_bias=True,
                                             kernel_initializer=tf.random_uniform_initializer(-stdv, stdv))
        # multi head
        self.in_att = [GraphAttentionLayer(self.conf.hidden_size, concat=True) for _ in range(self.conf.nb_heads)]
        self.out_att = GraphAttentionLayer(self.conf.hidden_size, concat=False)

        initializer = tf.initializers.GlorotUniform()
        self.w = tf.Variable(initializer(shape=[self.conf.nb_heads * self.conf.hidden_size, self.conf.hidden_size]), name="TRUST_PATH_w")

    def computer_somenode(self):
        ########  Node Attention initialization ########
        self.social_neighbors_values_input1 = tf.reduce_sum(
            input_tensor=tf.math.exp(
                self.first_low_att_layer_for_social_neighbors_layer1(tf.reshape(self.snii1, [-1, 1]))), axis=1)
        self.social_neighbors_values_input2 = tf.reduce_sum(
            input_tensor=tf.math.exp(
                self.second_low_att_layer_for_social_neighbors_layer1(tf.reshape(self.snii2, [-1, 1]))), axis=1)

        # ----------------------
        # user-item interest graph node attention initialization
        self.consumed_items_values_input1 = tf.reduce_sum(
            input_tensor=tf.math.exp(self.first_low_att_layer_for_user_item_layer1(tf.reshape(self.ciii1, [-1, 1]))),
            axis=1)
        self.consumed_items_values_input2 = tf.reduce_sum(
            input_tensor=tf.math.exp(self.second_low_att_layer_for_user_item_layer1(tf.reshape(self.ciii2, [-1, 1]))),
            axis=1)

        # ----------------------
        # item-user graph node attention initialization
        self.item_customer_values_input1 = tf.reduce_sum(
            input_tensor=tf.math.exp(self.first_low_att_layer_for_item_user_layer1(tf.reshape(self.icii1, [-1, 1]))),
            axis=1)
        self.item_customer_values_input2 = tf.reduce_sum(
            input_tensor=tf.math.exp(self.second_low_att_layer_for_item_user_layer1(tf.reshape(self.icii2, [-1, 1]))),
            axis=1)

        ######## Generate Sparse Matrices with/without attention #########
        # Frist Layer

        self.social_neighbors_sparse_matrix_avg = tf.SparseTensor(
            indices=self.social_neighbors_indices_input,
            values=self.social_neighbors_values_input,
            dense_shape=self.social_neighbors_dense_shape
        )
        self.first_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices=self.social_neighbors_indices_input,
            values=self.social_neighbors_values_input1,
            dense_shape=self.social_neighbors_dense_shape
        )
        self.consumed_items_sparse_matrix_avg = tf.SparseTensor(
            indices=self.consumed_items_indices_input,
            values=self.consumed_items_values_input,
            dense_shape=self.consumed_items_dense_shape
        )
        self.first_layer_consumed_items_sparse_matrix = tf.SparseTensor(
            indices=self.consumed_items_indices_input,
            values=self.consumed_items_values_input1,
            dense_shape=self.consumed_items_dense_shape
        )
        self.item_customer_sparse_matrix_avg = tf.SparseTensor(
            indices=self.item_customer_indices_input,
            values=self.item_customer_values_input,
            dense_shape=self.item_customer_dense_shape
        )
        self.first_layer_item_customer_sparse_matrix = tf.SparseTensor(
            indices=self.item_customer_indices_input,
            values=self.item_customer_values_input1,
            dense_shape=self.item_customer_dense_shape
        )
        # ----------------------
        # Second layer

        self.second_layer_social_neighbors_sparse_matrix = tf.SparseTensor(
            indices=self.social_neighbors_indices_input,
            values=self.social_neighbors_values_input2,
            dense_shape=self.social_neighbors_dense_shape
        )
        self.second_layer_consumed_items_sparse_matrix = tf.SparseTensor(
            indices=self.consumed_items_indices_input,
            values=self.consumed_items_values_input2,
            dense_shape=self.consumed_items_dense_shape
        )
        self.second_layer_item_customer_sparse_matrix = tf.SparseTensor(
            indices=self.item_customer_indices_input,
            values=self.item_customer_values_input2,
            dense_shape=self.item_customer_dense_shape
        )
        self.first_social_neighbors_low_level_att_matrix = tf.sparse.softmax(
            self.first_layer_social_neighbors_sparse_matrix)
        self.first_consumed_items_low_level_att_matrix = tf.sparse.softmax(
            self.first_layer_consumed_items_sparse_matrix)
        self.first_items_users_neighborslow_level_att_matrix = tf.sparse.softmax(
            self.first_layer_item_customer_sparse_matrix)
        self.second_social_neighbors_low_level_att_matrix = tf.sparse.softmax(
            self.second_layer_social_neighbors_sparse_matrix)
        self.second_consumed_items_low_level_att_matrix = tf.sparse.softmax(
            self.second_layer_consumed_items_sparse_matrix)
        self.second_items_users_neighborslow_level_att_matrix = tf.sparse.softmax(
            self.second_layer_item_customer_sparse_matrix)


    def call(self, user_input, item_input, labels_input, dataset, slice_indices, flag):
        if (flag == 0) or (flag == 1):
            self.computer_somenode()
            ########  Fusion Layer ########
            # modify
            self.fusion_item_embedding = self.item_embedding
            self.fusion_user_embedding = self.user_embedding[:-1]

            ######## Influence and Interest Diffusion Layer ########
            # ----------------------
            # First Layer
            user_embedding_from_consumed_items1 = self.generateUserEmebddingFromConsumedItems1(self.fusion_item_embedding)
            user_embedding_from_social_neighbors1 = self.generateUserEmbeddingFromSocialNeighbors1(self.fusion_user_embedding)

            consumed_items_attention1 = tf.math.exp(
                self.first_user_part_interest_graph_att_layer2(self.first_user_part_interest_graph_att_layer1( \
                    tf.concat([self.fusion_user_embedding, user_embedding_from_consumed_items1], 1)))) + 0.7
            social_neighbors_attention1 = tf.math.exp(
                self.first_user_part_social_graph_att_layer2(self.first_user_part_social_graph_att_layer1( \
                    tf.concat([self.fusion_user_embedding, user_embedding_from_social_neighbors1], 1)))) + 0.3

            sum_attention1 = consumed_items_attention1 + social_neighbors_attention1
            self.consumed_items_attention_1 = consumed_items_attention1 / sum_attention1
            self.social_neighbors_attention_1 = social_neighbors_attention1 / sum_attention1

            first_gcn_user_embedding = 1 / 2 * self.fusion_user_embedding \
                                       + 1 / 2 * (self.consumed_items_attention_1 * user_embedding_from_consumed_items1 \
                                                  + self.social_neighbors_attention_1 * user_embedding_from_social_neighbors1)

            item_itself_att1 = tf.math.exp(self.first_item_part_itself_graph_att_layer2( \
                self.first_item_part_itself_graph_att_layer1(self.fusion_item_embedding))) + 1.0

            item_customer_attenton1 = tf.math.exp(self.first_item_part_user_graph_att_layer2( \
                self.first_item_part_user_graph_att_layer1(
                    self.generateItemEmebddingFromCustomer1(self.fusion_user_embedding)))) + 1.0

            item_sum_attention1 = item_itself_att1 + item_customer_attenton1

            self.item_itself_att1 = item_itself_att1 / item_sum_attention1
            self.item_customer_attenton1 = item_customer_attenton1 / item_sum_attention1

            first_gcn_item_embedding = self.item_itself_att1 * self.fusion_item_embedding + self.item_customer_attenton1 * self.generateItemEmebddingFromCustomer1(
                self.fusion_user_embedding)

            # ----------------------
            # Second Layer
            user_embedding_from_consumed_items2 = self.generateUserEmebddingFromConsumedItems2(first_gcn_item_embedding)
            user_embedding_from_social_neighbors2 = self.generateUserEmbeddingFromSocialNeighbors2(first_gcn_user_embedding)

            consumed_items_attention2 = tf.math.exp(
                self.second_user_part_interest_graph_att_layer2(self.second_user_part_interest_graph_att_layer1( \
                    tf.concat([first_gcn_user_embedding, user_embedding_from_consumed_items2], 1)))) + 0.7

            social_neighbors_attention2 = tf.math.exp(
                self.second_user_part_social_graph_att_layer2(self.second_user_part_social_graph_att_layer1( \
                    tf.concat([first_gcn_user_embedding, user_embedding_from_social_neighbors2], 1)))) + 0.3

            sum_attention2 = consumed_items_attention2 + social_neighbors_attention2
            self.consumed_items_attention_2 = consumed_items_attention2 / sum_attention2
            self.social_neighbors_attention_2 = social_neighbors_attention2 / sum_attention2

            second_gcn_user_embedding = 1 / 2 * first_gcn_user_embedding \
                                        + 1 / 2 * (self.consumed_items_attention_2 * user_embedding_from_consumed_items2 \
                                                   + self.social_neighbors_attention_2 * user_embedding_from_social_neighbors2)

            second_mean_social_influ, second_var_social_influ = tf.nn.moments(x=self.social_neighbors_attention_2,
                                                                              axes=0)
            second_mean_interest_influ, second_var_interest_influ = tf.nn.moments(x=self.consumed_items_attention_2,
                                                                                  axes=0)
            self.second_layer_analy = [second_mean_social_influ, second_var_social_influ, \
                                       second_mean_interest_influ, second_var_interest_influ]

            item_itself_att2 = tf.math.exp(self.second_item_part_itself_graph_att_layer2( \
                self.second_item_part_itself_graph_att_layer1(first_gcn_item_embedding))) + 1.0

            item_customer_attenton2 = tf.math.exp(self.second_item_part_user_graph_att_layer2( \
                self.second_item_part_user_graph_att_layer1(
                    self.generateItemEmebddingFromCustomer2(first_gcn_user_embedding)))) + 1.0

            item_sum_attention2 = item_itself_att2 + item_customer_attenton2

            self.item_itself_att2 = item_itself_att2 / item_sum_attention2
            self.item_customer_attenton2 = item_customer_attenton2 / item_sum_attention2

            second_gcn_item_embedding = self.item_itself_att2 * first_gcn_item_embedding + self.item_customer_attenton2 * self.generateItemEmebddingFromCustomer2(first_gcn_user_embedding)

            ######## Prediction Layer ########
            self.final_user_embedding = tf.concat([first_gcn_user_embedding, second_gcn_user_embedding, self.user_embedding[:-1]], 1)
            self.final_item_embedding = tf.concat([first_gcn_item_embedding, second_gcn_item_embedding, self.item_embedding], 1)

            #---expert_s--------------------------------------------------------------------------------
            att1 = tf.nn.softmax(tf.matmul(self.final_user_embedding, self.att_exp1), 1)  # [B,2H] x [2H,2] = [B,2]
            self.fu = tf.multiply(first_gcn_user_embedding, tf.expand_dims(att1[:, 0], axis=1)) + tf.multiply(second_gcn_user_embedding, tf.expand_dims(att1[:, 1], axis=1)) + tf.multiply(self.user_embedding[:-1], tf.expand_dims(att1[:, 2], axis=1))

            att2 = tf.nn.softmax(tf.matmul(self.final_item_embedding, self.att_exp2), 1)  # [B,2H] x [2H,2] = [B,2]
            self.fi = tf.multiply(first_gcn_item_embedding, tf.expand_dims(att2[:, 0], axis=1)) + tf.multiply(second_gcn_item_embedding, tf.expand_dims(att2[:, 1], axis=1)) + tf.multiply(self.item_embedding, tf.expand_dims(att2[:, 2], axis=1))
            #------------------------------------------------------------------------------------

            latest_user_latent = tf.gather_nd(self.fu, user_input)
            latest_item_latent = tf.gather_nd(self.fi, item_input)
            predict_vector = tf.multiply(latest_user_latent, latest_item_latent)
            predict_score = tf.reduce_sum(input_tensor=predict_vector, axis=1)


            if flag == 1:
                return predict_score
            labels = tf.squeeze(tf.cast(labels_input, tf.float32), 1)

            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            loss1 = loss_fn(y_true=labels, y_pred=predict_score)
            loss1 = tf.reduce_mean(loss1)

        if (flag == 0) or (flag == 2):
            if flag == 0:
                path, mask, targets = dataset.get_slice(slice_indices)
            else:
                path, mask, targets, negs = dataset.get_slice(slice_indices)
            # multi head
            seq_l = tf.reduce_sum(mask, 1)  # path length
            mul_seq = tf.concat([att(self.user_embedding, path, seq_l) for att in self.in_att], 2)
            mul_seq_c = tf.concat([mul_seq[i] for i in range(mul_seq.get_shape()[0])], 0)
            mul_one = tf.matmul(mul_seq_c, self.w)
            mul_one = tf.nn.elu(mul_one)
            seq_hidden = self.out_att(self.user_embedding, tf.reshape(mul_one,
                                                                      [mul_seq.get_shape()[0], mul_seq.get_shape()[1],
                                                                       self.conf.hidden_size]), seq_l)

            last_ind = tf.stack([tf.range(mask.get_shape()[0]), tf.cast(seq_l - 1, dtype=tf.int32)], 1)
            ht = tf.gather_nd(seq_hidden, last_ind)  # batch_size x latent_size
            q1 = tf.expand_dims(self.linear_one(ht), 1)  # batch_size x 1 x latent_size
            q2 = self.linear_two(seq_hidden)  # batch_size x seq_length x latent_size
            alpha = self.linear_three(tf.sigmoid(q1 + q2))
            # [batch_size x seq_length x latent_size]
            a = tf.reduce_sum(alpha * seq_hidden * tf.reshape(tf.cast(mask, dtype=tf.float32), [mask.get_shape()[0], -1, 1]),1)  # float
            if not self.conf.nonhybrid:
                a = self.linear_transform(tf.concat([a, ht], 1))
            b = self.user_embedding[:-1]  # n_nodes x latent_size
            scores = tf.matmul(a, tf.transpose(b, perm=[1, 0]))

            if (flag == 2):
                return scores, negs
            loss2 = tf.nn.softmax_cross_entropy_with_logits(labels=tf.keras.utils.to_categorical(targets,
                                                                                                    num_classes=self.conf.num_users),
                                                               logits=scores)
            loss2 = tf.reduce_mean(loss2)
        return loss1,loss2




















