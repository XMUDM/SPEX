from __future__ import division
import os
from time import time
import tensorflow as tf
from Parser import parse_args
from utility.DataModule import DataModule
from utility.metrics import rec_test_single
from utility.Logging import Logging
from utility.Model import diffnetplus
args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id


def start(conf, d_train, d_test, model,optimizer):
    log_dir = os.path.join(os.getcwd(), 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(os.getcwd(), 'log/%s_rec.log' % (conf.dataset))
    log = Logging(log_path)
    # ----------------------
    data_dict = d_train.prepareModelSupplement(model)
    model.initializeNodes(data_dict)

    best_recall, best_ndcg, best_iter = [0, 0, 0], [0, 0, 0], [0, 0]
    # Start Training
    for epoch in range(1, conf.epochs+1):
        t0 = time()
        total_loss = 0
        d_train.parperTrainData()
        while d_train.terminal_flag:
            d_train.getTrainRankingBatch()
            #------------------------------------------------------------------
            with tf.GradientTape() as tape:
                predict_score, labels_input = model(d_train.user_list, d_train.item_list, d_train.labels_list, 0)
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_score, labels=labels_input))
                total_loss += loss.numpy()
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            # ------------------------------------------------------------------
        t1 = time()
        # print log to console and log_file
        log.record('Epoch:%d, Time:%.4fs, train loss:%.4f' % (epoch, (t1 - t0), total_loss))

        # ----------------------
        # evaluate model performance, recall and ndcg
        rec = rec_test_single(model, d_test)
        t2 = time()
        if rec["recall"][0] >= best_recall[0]:
            best_recall = rec["recall"]
            best_iter[0] = epoch
        if rec["ndcg"][0] >= best_ndcg[0]:
            best_ndcg = rec["ndcg"]
            best_iter[1] = epoch
        t3 = time()
        log.record('Evaluate cost:[%.4fs + %.4fs] recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % ((t2-t1),(t3-t2), rec["recall"][0], rec["recall"][1], rec["recall"][2], rec["ndcg"][0], rec["ndcg"][1], rec["ndcg"][2]))
    log.record('Best Result\n recall=[%.4f, %.4f, %.4f],  ndcg=[%.4f, %.4f, %.4f]' % (best_recall[0], best_recall[1], best_recall[2], best_ndcg[0], best_ndcg[1], best_ndcg[2]))


if __name__ == "__main__":
    d_train = DataModule(args, "%s%s/rec/%s.train.rating" % (args.data_root, args.dataset, args.dataset))
    d_test = DataModule(args, "%s%s/rec/%s.test.rating" % (args.data_root, args.dataset, args.dataset))
    d_train.initializeRankingTrain()
    d_test.initializeRankingTest()

    model = diffnetplus(args)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    start(args, d_train, d_test, model, optimizer)






