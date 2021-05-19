from __future__ import division
import tensorflow as tf
import numpy as np


class diffnetplus(tf.keras.Model):
    def __init__(self, conf):
        super(diffnetplus, self).__init__()
        self.conf = conf
        self.supply_set = (
            'SOCIAL_NEIGHBORS_SPARSE_MATRIX',
            'CONSUMED_ITEMS_SPARSE_MATRIX',
            'ITEM_CUSTOMER_SPARSE_MATRIX'
        )

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

        self.user_embedding = tf.Variable(tf.random.normal([self.conf.num_users, self.conf.hidden_size], stddev=0.01),
                                          name='user_embedding')
        self.item_embedding = tf.Variable(tf.random.normal([self.conf.num_items, self.conf.hidden_size], stddev=0.01),
                                          name='item_embedding')

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


        # -----nnnnnnnnn-------------------------------------------------------------------
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


    def call(self, user_input, item_input, labels_input, flag):
        if (flag == 0) or (flag == 1):
            self.computer_somenode()
            ########  Fusion Layer ########
            # modify
            self.fusion_item_embedding = self.item_embedding
            self.fusion_user_embedding = self.user_embedding

            ######## Influence and Interest Diffusion Layer ########
            # ----------------------
            # First Layer
            user_embedding_from_consumed_items1 = self.generateUserEmebddingFromConsumedItems1(
                self.fusion_item_embedding)
            user_embedding_from_social_neighbors1 = self.generateUserEmbeddingFromSocialNeighbors1(
                self.fusion_user_embedding)

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

            second_gcn_item_embedding = self.item_itself_att2 * first_gcn_item_embedding + \
                                        self.item_customer_attenton2 * self.generateItemEmebddingFromCustomer2(
                first_gcn_user_embedding)

            ######## Prediction Layer ########
            self.final_user_embedding = tf.concat(
                [first_gcn_user_embedding, second_gcn_user_embedding, self.user_embedding], 1)
            self.final_item_embedding = tf.concat(
                [first_gcn_item_embedding, second_gcn_item_embedding, self.item_embedding], 1)

            latest_user_latent = tf.gather_nd(self.final_user_embedding, user_input)
            latest_item_latent = tf.gather_nd(self.final_item_embedding, item_input)
            predict_vector = tf.multiply(latest_user_latent, latest_item_latent)
            predict_score = tf.reduce_sum(input_tensor=predict_vector, axis=1)

            if flag == 1:
                return predict_score
            return predict_score, tf.squeeze(tf.cast(labels_input, tf.float32), 1)


















