from __future__ import division
import os
import random
from time import time
import numpy as np
import tensorflow as tf
from Parser import parse_args
from utility.DataModule import DataModule
from utility.metrics import rec_test
from utility.Logging import Logging
from utility.Model_oy import MultiTaskModel
from utility2.Utils import loadData
from utility2.metrics import trust_test5

args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id


def start(conf, d_train, d_test, train_data, test_data, user_path_indx, alluser_path_index, model, optimizer):
    log_dir = os.path.join(os.getcwd(), 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # ----------------------
    # define log name
    log_path = os.path.join(os.getcwd(), 'log/%s_oy.log' % (conf.dataset))
    log = Logging(log_path)
    log.record(conf.num_users)
    log.record(conf.num_items)
    # ----------------------
    # prepare necessary data for diffnet++.
    print('System start to load graph...')
    data_dict = d_train.prepareModelSupplement(model)
    model.initializeNodes(data_dict)
    model.initializeNodes2()

    # rec
    best_recall, best_ndcg, best_iter = [0, 0, 0], [0, 0, 0], [0, 0]
    # trust
    bestrecall, bestndcg = [0, 0, 0], [0, 0, 0]

    # Start Training
    for epoch in range(1, conf.epochs + 1):
        ################ train ####################
        t0 = time()
        total_loss1 = 0
        total_loss2 = 0
        d_train.parperTrainData()  # 每个epoch进行负采样
        while d_train.terminal_flag:
            d_train.getTrainRankingBatch()  # 获得一个batch数据
            trust_batch_size = len(alluser_path_index) // (len(d_train.all_user_list) // conf.batch_size)

            unique_user = set(np.ravel(d_train.user_list).tolist())
            path_index = []
            for u in unique_user:
                path_index.extend(user_path_indx[u])
            # if len(path_index) > len(unique_user):
            #     path_index = random.sample(path_index, len(unique_user))
            if len(path_index) > trust_batch_size:
                path_index = random.sample(path_index, trust_batch_size)

            # -------------------------------------------------------------------------------------
            with tf.GradientTape() as tape:
                loss1, loss2 = model(d_train.user_list, d_train.item_list, d_train.labels_list, train_data, path_index, 0)
                total_loss1 += loss1.numpy()
                total_loss2 += loss2.numpy()

                # multi
                T = len(path_index)
                n_rec = 5
                T_rec = len(d_train.user_list)
                precision1 = tf.math.exp(-2 * model.task_weights[0])
                precision2 = tf.math.exp(-2 * model.task_weights[1])
                loss = precision1 * loss1 + precision2 * loss2 + 2 * (n_rec + 1) * T_rec * model.task_weights[0] + T * model.task_weights[1]
                # log.record('%d,%.5f,%.5f,%.5f,%.5f' % (epoch, precision1.numpy(), precision2.numpy(), total_loss1, total_loss2))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # -------------------------------------------------------------------------------------
        t1 = time()
        log.record('%d,%.5f,%.5f,%.5f,%.5f,Time[%.4f]' % (
        epoch, precision1.numpy(), precision2.numpy(), total_loss1, total_loss2, t1 - t0))

        ################ test rec ####################
        rec = rec_test(model, d_test)
        t2 = time()
        log.record('epoch[%d], Recall=[%.5f, %.5f, %.5f], NDCG=[%.5f, %.5f, %.5f], Time:[%.5f]' % (
        epoch, rec["recall"][0], rec["recall"][1], rec["recall"][2], rec["ndcg"][0], rec["ndcg"][1], rec["ndcg"][2],
        (t2 - t1)))
        if rec["recall"][0] > best_recall[0]:
            best_recall = rec["recall"]
            best_iter[0] = epoch
        if rec["ndcg"][0] > best_ndcg[0]:
            best_ndcg = rec["ndcg"]
            best_iter[1] = epoch

        ################ test trust ####################
        recall10, recall20, recall50, ndcg10, ndcg20, ndcg50 = trust_test5(model, test_data)
        t3 = time()
        log.record('epoch[%d], Recall[%.5f,%.5f,%.5f], NDCG:[%.5f,%.5f,%.5f], Time:[%.5f]' % (
        epoch, recall10, recall20, recall50, ndcg10, ndcg20, ndcg50, t3 - t2))
        if recall10 > bestrecall[0]:
            bestrecall = [recall10, recall20, recall50]
        if ndcg10 > bestndcg[0]:
            bestndcg = [ndcg10, ndcg20, ndcg50]
    log.record('Best REC: recall=[%.5f, %.5f, %.5f],  ndcg=[%.5f, %.5f, %.5f]' % (
        best_recall[0], best_recall[1], best_recall[2], best_ndcg[0], best_ndcg[1], best_ndcg[2]))
    log.record('Best TRUST: Recall[%.5f,%.5f,%.5f], NDCG:[%.5f,%.5f,%.5f]' % (
        bestrecall[0], bestrecall[1], bestrecall[2], bestndcg[0], bestndcg[1], bestndcg[2]))


if __name__ == "__main__":
    # rec data
    d_train = DataModule(args, "%s%s/rec/%s.train.rating" % (args.data_root, args.dataset, args.dataset))
    d_test = DataModule(args, "%s%s/rec/%s.test.rating" % (args.data_root, args.dataset, args.dataset))
    d_train.initializeRankingTrain()
    d_test.initializeRankingTest()
    # trust data
    train_data, test_data, user_path_indx, alluser_path_index = loadData(args)
    model = MultiTaskModel(args)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    start(args, d_train, d_test, train_data, test_data, user_path_indx, alluser_path_index, model, optimizer)
