import os
import time
import datetime
import random
import argparse
from sklearn.metrics import log_loss, roc_auc_score
import tensorflow as tf
import numpy as np

from utils import *
from DIR import *


def eval(args, model, dataset, isrank, split_hist, union_hist, _print=False):
    preds, labels, losses, masks = [], [], [], []

    batch_num = len(dataset[0]) // args.batch_size
    print('eval', args.batch_size, batch_num)

    t = time.time()
    for batch_no in range(batch_num):
        batch = get_aggregated_batch(dataset, args.batch_size, batch_no)
        pred, label, mask, loss = model.eval(batch, split_hist, union_hist, not batch_no)
        preds.extend(pred)
        losses.append(loss)
        labels.extend(label)
        masks.extend(mask)
    loss = sum(losses) / len(losses)
    res = evaluate(preds, labels, args.cate_num, args.list_len, args.metric_scopes,
                   args.expo_ratio, args.expo_scope, isrank, _print)

    masks = np.nonzero(np.array(masks).reshape(-1))[0]
    auc = roc_auc_score(np.array(labels).reshape(-1)[masks], np.array(preds).reshape(-1)[masks])
    print("EVAL TIME: %.4fs" % (time.time() - t))
    return loss, res, auc



def train(args):
    tf.reset_default_graph()
    print('algo: DIR', 'lr', args.lr, 'l2', args.l2_norm)
    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)
    split_hist, split_candi, cloud_hist, reverse_hist = True, True, True, False

    loss = 'STE'
    model = DIR(args, loss)
    union_hist = True

    with model.graph.as_default() as g:
        sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.set_sess(sess)

    # train_set = pkl.load(open(args.train_data_dir, 'rb'))
    train_set = load_data(args.train_data_dir, args.cate_idx_map, args.cate_idx, args.list_len, args.union_hist_len,
                          args.cate_hist_len, split_candi, split_hist, cloud_hist, reverse_hist, union_hist)
    print('finish loading training set')
    test_set = load_data(args.test_data_dir, args.cate_idx_map, args.cate_idx, args.list_len, args.union_hist_len,
                          args.cate_hist_len, split_candi, split_hist, cloud_hist, reverse_hist, union_hist)
    print('finish loading test set')

    # test_set = pkl.load(open(args.test_data_dir, 'rb'))

    model_name = '{}_DIR_{}_{}_{}_{}_{}_{}_{}_{}_vec2'.format(args.timestamp, args.batch_size,
        args.lr, args.l2_norm, args.emb_dim, args.keep_prob, args.hidd_size, args.expert_num, args.n_head)
    if not os.path.exists('{}/{}/'.format(args.save_dir, model_name)):
        os.makedirs('{}/{}/'.format(args.save_dir, model_name))
    save_path = '{}/{}/ckpt'.format(args.save_dir, model_name)
    metrics_save_path = '{}/{}.metrics'.format(args.save_dir, model_name)

    training_monitor = {
        'train_loss': [],
        'vali_loss': [],
        'utility': [],
        'ndcg': [],
        'map': [],
        'JS': [],
        'auc': [],
    }

    # gpu settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    # training process
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_losses_step = []

        # before training process
        step = 0
        vali_loss, res, auc = eval(args, model, test_set, isrank=False, split_hist=split_hist, union_hist=union_hist)

        training_monitor['train_loss'].append(None)
        training_monitor['vali_loss'].append(vali_loss)

        print("STEP %d  LOSS TRAIN: NULL | LOSS VALI: %.4f  AUC: %.4f JS: %.4f" % (step, vali_loss, auc, res[0]))
        for i, scope in enumerate(args.metric_scopes):
            print("\t@%d  MAP: %.4f  NDCG: %.4f  CLICK: %.4f" % (scope, res[1][i], res[2][i], res[3][i]))
        early_stop = False
        batch_num = len(train_set[0]) // args.batch_size
        eval_iter_num = batch_num // args.eval_freq if batch_num > 50 else batch_num
        best_util = 0
        print('train', batch_num, 'eval iter num', eval_iter_num)

        # begin training process
        for epoch in range(args.epoch_num):
            # if early_stop:
            #     break
            for batch_no in range(batch_num):
                batch = get_aggregated_batch(train_set, args.batch_size, batch_no)
                loss = model.train(batch, split_hist=split_hist, union_hist=union_hist)

                step += 1
                train_losses_step.append(loss)

                if step % eval_iter_num == 0:
                    train_loss = sum(train_losses_step) / len(train_losses_step)
                    training_monitor['train_loss'].append(train_loss)
                    train_losses_step = []

                    vali_loss, res, auc = eval(args, model, test_set, isrank=True, split_hist=split_hist, union_hist=union_hist)
                    training_monitor['vali_loss'].append(vali_loss)
                    training_monitor['utility'].append(res[0])

                    print("EPOCH %d  STEP %d  LOSS TRAIN: %.4f | LOSS VALI: %.4f   AUC: %.4f JS: %.4f" % (epoch,
                                                    step, train_loss, vali_loss, auc, res[0]))
                    for i, scope in enumerate(args.metric_scopes):
                        print("\t@%d  MAP: %.4f  NDCG: %.4f  CLICK: %.4f" % (scope, res[1][i], res[2][i], res[3][i]))
                    if res[0] > best_util:
                        # save model
                        best_util = res[0]
                        model.save(save_path)
                        pkl.dump(res[-1], open(metrics_save_path, 'wb'))
                        print('model saved')
                        early_stop = False
                        continue

                    if len(training_monitor['utility']) > 2 and epoch > 0:
                        if epoch > 40 and best_util - training_monitor['utility'][-1] > 0.08:
                            early_stop = True
                        # if (training_monitor['vali_loss'][-1] > training_monitor['vali_loss'][-2] and
                        #         training_monitor['vali_loss'][-2] > training_monitor['vali_loss'][-3]):
                        #     early_stop = True
                        # if (training_monitor['ctr'][-2] - training_monitor['ctr'][-1]) >= 0.1 and (
                        #         training_monitor['ctr'][-3] - training_monitor['ctr'][-2]) >= 0.1:
                        #     early_stop = True


def reranker_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', default='../data/taobao/process/data.train')
    parser.add_argument('--test_data_dir', default='../data/taobao/process/data.test')
    parser.add_argument('--stat_dir', default='../data/taobao/process/data.stat')
    parser.add_argument('--save_dir', default='../model/taobao/')
    parser.add_argument('--expo_ratio', default=[0.5, 0.4, 0.1], type=list, help='the ratio of exposure')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    parser.add_argument('--epoch_num', default=400, type=int, help='epochs of each iteration.') # tb
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--cate_hist_len', default=20, type=int, help='the max length of history for each category')
    parser.add_argument('--union_hist_len', default=30, type=int, help='the max length of union history')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--l2_norm', default=1e-4, type=float, help='l2 loss scale')
    parser.add_argument('--js_lambda', default=0.5, type=float, help='js lambda')
    parser.add_argument('--keep_prob', default=0.8, type=float, help='keep probability')
    parser.add_argument('--emb_dim', default=16, type=int, help='size of embedding')
    parser.add_argument('--hidd_size', default=32, type=int, help='hidden size')
    parser.add_argument('--d_model', default=64, type=int, help='input dimension of FFN')
    parser.add_argument('--d_inner_hid', default=128, type=int, help='hidden dimension of FFN')
    parser.add_argument('--n_head', default=4, type=int, help='the number of head in self-attention')
    parser.add_argument('--expert_num', default=4, type=int, help='the number of expert in MMoE')
    parser.add_argument('--hidden_layer', default=[256, 128, 64], type=int, help='size of hidden layer')
    parser.add_argument('--final_layers_arch', default=[200, 80], type=int, help='size of final layer')
    parser.add_argument('--metric_scopes', default=[1, 5, 7], type=list, help='the scope of metrics')
    parser.add_argument('--expo_scope', default=5, type=int, help='the scope of exposure')
    parser.add_argument('--grad_norm', default=0, type=float, help='max norm of gradient')
    parser.add_argument('--decay_steps', default=3000, type=int, help='learning rate decay steps')
    parser.add_argument('--decay_rate', default=1.0, type=float, help='learning rate decay rate')
    parser.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%Y%m%d%H%M"))
    parser.add_argument('--reload_path', type=str, default='', help='model ckpt dir')
    parser.add_argument('--setting_path', type=str, default='', help='setting dir')
    parser.add_argument('--eval_freq', type=int, default=4,
                               help='the frequency of evaluating on the valid set when training')

    args, _ = parser.parse_known_args()

    with open(args.stat_dir, 'r') as f:
        stat = json.load(f)

    args.feat_size = stat['ft_num']
    args.cate_num = stat['cate_num']
    args.list_len = stat['list_len']
    # print('list len', args.list_len)
    args.union_hist_len = stat['hist_len']
    # args.cate_hist_len = stat['hist_cate_len']
    args.cate_idx_map = stat['cate_idx_map']
    print('cate idx map', args.cate_idx_map)
    args.cate_idx = stat['cate_idx']
    args.itm_fnum = stat['itm_fnum']
    args.itm_dens_fnum = stat.get('itm_dens_ft', 0)
    args.usr_fnum = stat['usr_fnum']
    args.hist_fnum = stat['hist_fnum']
    args.hist_dens_fnum = stat.get('hist_dens_ft', 0)
    print('itm dens fnum', args.itm_dens_fnum, 'hist dens fnum', args.hist_dens_fnum)

    return args


if __name__ == '__main__':
    # parameters
    args = reranker_parse_args()
    print(args.timestamp)
    if args.setting_path:
        parse = load_parse_from_json(args, args.setting_path)
    # set seed
    random.seed(args.seed)
    tf.random.set_random_seed(args.seed)
    np.random.seed(args.seed)
    train(args)
