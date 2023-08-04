import abc

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, static_bidirectional_rnn, LSTMCell
import numpy as np


class BaseModel(object):
    def __init__(self, args):
        self.args = args
        self.feat_size = args.feat_size
        self.cate_num = args.cate_num
        self.list_len = args.list_len
        self.union_hist_len = args.union_hist_len
        self.cate_hist_len = args.cate_hist_len
        self.itm_fnum = args.itm_fnum
        self.itm_dens_fnum = args.itm_dens_fnum
        self.usr_fnum = args.usr_fnum
        self.hist_fnum = args.hist_fnum
        self.hist_dens_fnum = args.hist_dens_fnum
        # self.cate_idx = args.cate_idx
        self.emb_dim = args.emb_dim
        self.hidden_size = args.hidd_size
        self.max_grad_norm = args.grad_norm
        self.l2_norm = args.l2_norm
        self.lr = args.lr
        self.kp = args.keep_prob
        self.dens_emb_dim = self.hidden_size*3 if self.itm_dens_fnum else 0
        self.itm_ft_dim = self.itm_fnum * self.emb_dim + self.dens_emb_dim
        self.hst_ft_dim = self.hist_fnum * self.emb_dim + self.dens_emb_dim
        self.final_layers_arch = args.final_layers_arch
        # reset graph
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.construct_input()
            self.construct_network()

    def construct_input(self):
        # input placeholders
        with tf.name_scope('inputs'):
            self.itm_ft_ph = tf.placeholder(tf.int32, [None, self.cate_num, self.list_len, self.itm_fnum], name='candi_itm_ft')
            if self.itm_dens_fnum > 0:
                self.itm_dens_ph = tf.placeholder(tf.float32, [None, self.cate_num, self.list_len, self.itm_dens_fnum], name='candi_itm_dens')

            self.lb_ph = tf.placeholder(tf.float32, [None, self.cate_num, self.list_len], name='lb')
            self.candi_len_ph = tf.placeholder(tf.int32, [None, self.cate_num], name='candi_list_len')
            self.usr_ph = tf.placeholder(tf.int32, [None, self.usr_fnum], name='usr_ph')
            self.union_hist_ph = tf.placeholder(tf.int32, [None, self.union_hist_len, self.hist_fnum], name='union_hist_ft')
            self.union_hist_len_ph = tf.placeholder(tf.int32, [None], name='union_hist_len')
            self.cate_hist_ph = tf.placeholder(tf.int32, [None, self.cate_num, self.cate_hist_len, self.hist_fnum], name='cate_hist_ft')
            self.cate_hist_len_ph = tf.placeholder(tf.int32, [None, self.cate_num], name='cate_hist_len')
            if self.hist_dens_fnum > 0:
                self.union_hist_dens_ph = tf.placeholder(tf.float32, [None, self.union_hist_len, self.hist_dens_fnum], name='union_hist_dens')
                self.cate_hist_dens_ph = tf.placeholder(tf.float32, [None, self.cate_num, self.cate_hist_len, self.hist_dens_fnum], name='cate_hist_dens')

            self.is_train = tf.placeholder(tf.bool, [], name='is_train')

            # keep prob
            self.keep_prob = tf.placeholder(tf.float32, [])

        # embedding
        with tf.name_scope('embedding'):
            self.emb_mtx = tf.get_variable('emb_mtx', [self.feat_size + 1, self.emb_dim],
                                           initializer=tf.truncated_normal_initializer)
            self.block_emb = tf.get_variable('cate_emb', [self.cate_num, self.emb_dim],
                                             initializer=tf.truncated_normal_initializer)
            self.candi_emb = tf.gather(self.emb_mtx, self.itm_ft_ph)
            self.candi_emb = tf.reshape(self.candi_emb, [-1, self.cate_num,
                                                        self.list_len, self.itm_fnum * self.emb_dim])
            self.usr_emb = tf.gather(self.emb_mtx, self.usr_ph)
            self.usr_emb = tf.reshape(self.usr_emb, [-1, 1, 1, self.usr_fnum * self.emb_dim])
            self.cate_hist_emb = tf.gather(self.emb_mtx, self.cate_hist_ph)
            self.cate_hist_emb = tf.reshape(self.cate_hist_emb, [-1, self.cate_num, self.cate_hist_len,
                                                            self.hist_fnum * self.emb_dim])
            self.union_hist_emb = tf.gather(self.emb_mtx, self.union_hist_ph)
            self.union_hist_emb = tf.reshape(self.union_hist_emb, [-1, self.union_hist_len,
                                                            self.hist_fnum * self.emb_dim])

        if self.itm_dens_fnum > 0:
            itm_dens_emb = tf.layers.dense(self.itm_dens_ph, self.dens_emb_dim, name='emb_fc')
            self.candi_emb = tf.concat([self.candi_emb, itm_dens_emb], axis=-1)
        if self.hist_dens_fnum > 0:
            union_hist_dens_emb = tf.layers.dense(self.union_hist_dens_ph, self.dens_emb_dim, name='emb_fc', reuse=True)
            self.union_hist_emb = tf.concat([self.union_hist_emb, union_hist_dens_emb], axis=-1)
            cate_hist_dens_emb = tf.layers.dense(self.cate_hist_dens_ph, self.dens_emb_dim, name='emb_fc', reuse=True)
            self.cate_hist_emb = tf.concat([self.cate_hist_emb, cate_hist_dens_emb], axis=-1)

        cate_hist_mask = tf.sequence_mask(self.cate_hist_len_ph, maxlen=self.cate_hist_len, dtype=tf.float32)
        self.cate_hist_mask = tf.expand_dims(cate_hist_mask, axis=-1)
        self.cate_hist_emb *= self.cate_hist_mask

        tile_user = tf.tile(self.usr_emb, [1, self.cate_num, self.list_len, 1])
        self.tile_user = tf.reshape(tile_user, [-1, self.cate_num, self.list_len, self.usr_fnum * self.emb_dim])
        mask = tf.sequence_mask(self.candi_len_ph, maxlen=self.list_len, dtype=tf.float32)
        self.candi_mask = tf.expand_dims(mask, axis=-1)
        self.candi_emb = self.candi_emb * self.candi_mask
        self.mask = tf.reshape(mask, [-1, self.cate_num * self.list_len])
        self.labels = tf.reshape(self.lb_ph, [-1, self.cate_num * self.list_len])


    @abc.abstractmethod
    def construct_network(self):
        return

    def final_pred_net(self, inp, layer=(200, 80), poso_weight=None, fin_act='sigmoid', scope='mlp'):
        shape = inp.shape
        inp = tf.reshape(inp, [-1, shape[-1]])
        tile_num = np.prod(shape[1:-1])
        # print('tile num', tile_num)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            inp = tf.layers.batch_normalization(inputs=inp, name='mlp_bn', training=self.is_train)
            for i, hidden_num in enumerate(layer):
                fc = tf.layers.dense(inp, hidden_num, activation=tf.nn.relu, name='fc' + str(i))
                if poso_weight and i == len(layer)-1:
                    weights = tf.tile(tf.expand_dims(poso_weight[0], 1), [1, tile_num, 1])
                    weights = tf.reshape(weights, [-1, hidden_num])
                    # print('----------------------', fc.shape, weights.shape, shape)
                    fc = fc * weights
                # bn = tf.layers.batch_normalization(inputs=fc, name='bn' + str(i), training=self.is_train)
                inp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
            if fin_act == 'sigmoid':
                final = tf.layers.dense(inp, 1, activation=tf.nn.sigmoid, name='fc_final')
                score = tf.reshape(final, [-1, shape[-2]])
            elif fin_act == 'softmax':
                final = tf.layers.dense(inp, 1, activation=None, name='fc_final')
                score = tf.nn.softmax(tf.reshape(final, [-1, shape[-2]]))
            # seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            # y_pred = seq_mask * score
        return score

    def build_logloss(self, y_preds, lbs):
        # loss
        self.loss = tf.losses.log_loss(lbs, y_preds)
        self.opt()

    def opt(self):
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.l2_norm * tf.nn.l2_loss(v)

        self.optimizer = tf.train.AdamOptimizer(self.lr)

        if self.max_grad_norm > 0:
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            for idx, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
            self.train_step = self.optimizer.apply_gradients(grads_and_vars)
        else:
            self.train_step = self.optimizer.minimize(self.loss)

    def multihead_attention(self, queries, keys, values, num_units=None, num_heads=2,
                            scope="multihead_att", reuse=None, bias=False, cross=False):
        with tf.variable_scope(scope, reuse=reuse):
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            V = tf.layers.dense(values, num_units, activation=None)  # (N, T_k, C)
            if cross:
                Q = tf.stack(self.orig_cross_stitch(tf.unstack(Q, axis=1), 'q_cs'), axis=1)
                K = tf.stack(self.orig_cross_stitch(tf.unstack(K, axis=1), 'q_cs'), axis=1)
                V = tf.stack(self.orig_cross_stitch(tf.unstack(V, axis=1), 'q_cs'), axis=1)

            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            if bias:
                q_len = queries.get_shape().as_list()[1]
                k_len = queries.get_shape().as_list()[1]
                w = tf.get_variable("w", [q_len, k_len], initializer=tf.truncated_normal_initializer)
                outputs = outputs + w

            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)
            outputs = tf.nn.dropout(outputs, self.keep_prob)
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        return outputs

    def orig_cross_stitch(self, representations, scope='ocs'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            num = len(representations)
            inp = tf.stack(representations, axis=-1)
            cs_shape = (len(representations[0].shape) - 2) * [1] + [num, num]
            cross_stitch = tf.get_variable('cross_stitch', shape=cs_shape,
                                           dtype=tf.float32, initializer=tf.initializers.identity())
            out = tf.matmul(inp, cross_stitch)
            out = tf.unstack(out, axis=-1)
        return out

    def prepare_feed_data(self, batch_data, split_hist, union_hist):
        feed_dict = {
            self.usr_ph: np.array(batch_data[0]).reshape([-1, self.usr_fnum]),
            self.itm_ft_ph: np.array(batch_data[1])[:, :, :, :self.itm_fnum],
            self.candi_len_ph: batch_data[2],
            self.lb_ph: batch_data[3],
        }
        if self.itm_dens_fnum > 0:
            feed_dict[self.itm_dens_ph] = np.array(batch_data[1])[:, :, :, self.itm_fnum:]
        if split_hist:
            feed_dict[self.cate_hist_ph] = np.array(batch_data[4])[:, :, :, :self.hist_fnum]
            feed_dict[self.cate_hist_len_ph] = batch_data[5]
        if union_hist:
            feed_dict[self.union_hist_ph] = np.array(batch_data[6])[:, :, :self.hist_fnum]
            feed_dict[self.union_hist_len_ph] = batch_data[7]
        if self.hist_dens_fnum > 0:
            if split_hist:
                feed_dict[self.cate_hist_dens_ph] = np.array(batch_data[4])[:, :, :, self.hist_fnum:]
            if union_hist:
                feed_dict[self.union_hist_dens_ph] = np.array(batch_data[6])[:, :, self.hist_fnum:]
        return feed_dict

    def train(self, batch_data, split_hist, union_hist):
        feed_dict = self.prepare_feed_data(batch_data, split_hist, union_hist)
        feed_dict[self.keep_prob] = self.kp
        feed_dict[self.is_train] = True

        with self.graph.as_default():
            loss, _ = self.sess.run([self.loss, self.train_step], feed_dict=feed_dict)
            return loss

    def eval(self, batch_data, split_hist, union_hist, no_print=False):
        feed_dict = self.prepare_feed_data(batch_data, split_hist, union_hist)
        feed_dict[self.keep_prob] = 1.0
        feed_dict[self.is_train] = False

        with self.graph.as_default():
            pred, loss, lb, mask = self.sess.run([self.preds, self.loss, self.labels, self.mask], feed_dict=feed_dict)
            return pred, lb, mask, loss

    def save(self, path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, save_path=path)
            print('Save model:', path)

    def load(self, path):
        with self.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(path)
            if ckpt and ckpt.model_checkpoint_path:
                saver = tf.train.Saver()
                saver.restore(sess=self.sess, save_path=ckpt.model_checkpoint_path)
                print('Restore model:', ckpt.model_checkpoint_path)

    def set_sess(self, sess):
        self.sess = sess


class DIR(BaseModel):
    def __init__(self, args, loss):
        super(DIR, self).__init__(args)
        print('construct', 'loss', loss)
        with self.graph.as_default():
            if loss == 'll':
                self.build_logloss(self.preds, self.labels)
            else:
                self.build_js_loss(self.preds, self.labels, args.expo_scope, args.expo_ratio, mode=loss)

    def construct_network(self):
        self.gru_poso()

        # single tower
        preds = self.final_pred_net(self.final_emb, self.final_layers_arch, poso_weight=self.poso_weight)
        # multiple towers
        # final_emb_dim = self.final_emb.get_shape().as_list()[-1]
        # cate_embs = tf.unstack(tf.reshape(self.final_emb, [-1, self.cate_num, self.list_len, final_emb_dim]), axis=1)
        # preds = [self.final_pred_net(cate_emb, self.final_layers_arch, self.poso_weight,
        #                              scope='final_mlp_'+str(i)) for i, cate_emb in enumerate(cate_embs)]
        # preds = tf.stack(preds, axis=1)

        self.preds = tf.reshape(preds, [-1, self.cate_num * self.list_len]) * self.mask


    def gru_poso(self):
        cate_hist_emb = tf.unstack(self.cate_hist_emb, axis=1)
        cate_hist_len = tf.unstack(self.cate_hist_len_ph, axis=1)
        # with gru
        gru_cell = GRUCell(self.hidden_size, name='cell')
        hist_gru = [tf.nn.dynamic_rnn(gru_cell, cate_hist_emb[i], cate_hist_len[i], dtype='float32')[1] for i in range(self.cate_num)]

        # with cross stitch
        cross_hist = self.orig_cross_stitch(hist_gru, 'hist_pool_cross')
        _cross_hist = tf.stack(cross_hist, axis=1)
        stack_gru = tf.stack(hist_gru, axis=1)
        cross_hist = tf.concat([_cross_hist, stack_gru], axis=-1)

        fc_c = tf.layers.dense(self.candi_emb, self.hidden_size, activation=tf.nn.relu, name='fc_c')
        # fc_c = tf.layers.dense(self.candi_emb, self.hidden_size, activation=None, name='fc_c')

        cross_shape = cross_hist.shape[-1]
        fc_h = tf.reshape(cross_hist, [-1, self.cate_num, 1, cross_shape])
        fc_h_tile = tf.tile(fc_h, [1, 1, self.list_len, 1])
        concat_emb = tf.reshape(fc_h_tile, [-1, self.cate_num * self.list_len, cross_shape])
        # HSA
        emb = self.multihead_attention(concat_emb, concat_emb, concat_emb, num_units=self.hidden_size)

        emb = tf.reshape(emb, [-1, self.cate_num, self.list_len, self.hidden_size])

        poso_emb = tf.reduce_mean(emb, axis=-2)
        poso_emb = tf.reshape(poso_emb, [-1, self.cate_num * self.hidden_size])
        self.poso_weight = [tf.layers.dense(poso_emb, self.final_layers_arch[-1], activation=tf.nn.sigmoid, name='fc')]

        # with candidate SA
        candi_emb = tf.reshape(self.candi_emb, [-1, self.cate_num * self.list_len, self.itm_ft_dim])
        fc_att = self.multihead_attention(candi_emb, candi_emb, candi_emb,
                                          num_units=self.hidden_size, scope='fc_att')
        fc_att = tf.reshape(fc_att, [-1, self.cate_num, self.list_len, self.hidden_size])
        self.final_emb = tf.concat([fc_h_tile, fc_att, fc_att * emb], axis=-1)
        # self.final_emb = tf.concat([fc_h_tile, fc_c, fc_att, fc_c * emb], axis=-1)


    def build_js_loss(self, preds, labels, exposure_num, ideal_ratio, tau=1.0, mode='gumbel_topk'):
        # cross entropy
        ll_loss = tf.losses.log_loss(labels, preds)
        if mode == 'gumbel_softmax':
            pick = self.gumbel_softmax(preds, exposure_num, tau)
        elif mode == 'gumbel_STE':
            pick = self.gumbel_STE(preds, exposure_num)
        else:
            pick = self.STE(preds, exposure_num)

        total_pick = tf.reduce_sum(tf.reduce_sum(pick))
        cate_pick = tf.split(pick, self.cate_num, axis=-1)
        cate_total_pick = [tf.reduce_sum(tf.reduce_sum(cate_pick[i])) for i in range(self.cate_num)]
        batch_ratio = [tf.cast(num, tf.float32) / tf.cast(total_pick, tf.float32) for num in cate_total_pick]
        batch_ratio = tf.stack(batch_ratio)
        ideal_ratio = tf.convert_to_tensor(ideal_ratio, tf.float32, name='global_mask')
        M_ratio = (batch_ratio + ideal_ratio) / 2
        js_loss = 0.5 * self.KL_divergence(batch_ratio, M_ratio) + 0.5 * self.KL_divergence(ideal_ratio, M_ratio)
        print('------------------js lambda', self.args.js_lambda)
        self.loss = self.args.js_lambda * ll_loss * 2 + (1 - self.args.js_lambda) * js_loss * 2
        self.opt()

    def STE(self, scores, k):
        value, idx = tf.nn.top_k(scores, k)
        onehot = tf.one_hot(idx, self.cate_num * self.list_len)
        khot = tf.reduce_sum(onehot, axis=-2)
        khot = tf.cast(khot, tf.float32)
        pick = tf.stop_gradient(khot - scores) + scores
        return pick

    def gumbel_STE(self, scores, k):
        scores = scores + self.sample_gumbel_01(tf.shape(scores))
        return self.STE(scores, k)

    def gumbel_softmax(self, scores, k,  tau=1.0):
        scores = scores + self.sample_gumbel_01(tf.shape(scores))
        khot = tf.zeros_like(scores)
        onehot_approx = tf.zeros_like(scores)
        for i in range(k):
            khot_mask = tf.maximum(1.0 - onehot_approx, -2 ** 32 + 1)
            scores += tf.log(khot_mask)
            onehot_approx = tf.nn.softmax(scores / tau, dim=-1)
            khot += onehot_approx
        return self.STE(khot, k)

    def sample_gumbel_01(self, shape, eps=1e-10):
        # sample from gumbel(0, 1) distribution
        U = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32)
        return -tf.math.log(-tf.math.log(U + eps) + eps)

    def KL_divergence(self, p, q):
        return tf.reduce_sum(p * tf.log(p / (q + 1e-9)), axis=-1)


