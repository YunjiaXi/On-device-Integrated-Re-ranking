import json
from collections import defaultdict
import pickle as pkl
import numpy as np
import scipy.stats


def KL_divergence(p, q):
    p, q = np.array(p), np.array(q)
    return scipy.stats.entropy(p, q)


def JS_divergence(p, q):
    p, q = np.array(p), np.array(q)
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


def get_batch(data, batch_size, batch_no):
    return data[batch_size * batch_no: batch_size * (batch_no + 1)]


def load_parse_from_json(parse, setting_path):
    with open(setting_path, 'r') as f:
        setting = json.load(f)
    parse_dict = vars(parse)
    for k, v in setting.items():
        parse_dict[k] = v
    return parse


def get_aggregated_batch(data, batch_size, batch_no):
    return [data[d][batch_size * batch_no: batch_size * (batch_no + 1)] for d in range(len(data))]


# 需要把历史的顺序处理成用户先点击的在前面，最近点击的在后面的形式
def load_data(data_path, cate_idx_map, cate_idx, candi_max_len, union_hist_len, cate_hist_len=None, split_candi=True,
              split_hist=True, cloud_hist=False, reverse_hist=False, union_hist=True):
    cate_num = len(cate_idx_map)
    usr_ft_lists, candi_lists, candi_lb_lists, hist_lists, hist_len_lists = pkl.load(open(data_path, 'rb'))
    proc_candi_itms, proc_lbs, proc_candi_lens = [], [], []
    full_candi_itms = []
    if split_candi:
        for candi_itms, candi_lbs in zip(candi_lists, candi_lb_lists):
            if sum(candi_lbs) == 0:
                print('-----------------')
            candi_seq_len = len(candi_itms)
            candi_seq = list(candi_itms[:candi_max_len]) + [[0 for _ in range(len(candi_itms[0]))] for _ in
                                                            range(candi_max_len - candi_seq_len)]
            full_candi_itms.append(candi_seq)
            candi_seq, candi_lb_seq, candi_len_seq = [[] for _ in range(cate_num)], [[] for _ in range(cate_num)], []
            for itm, lb in zip(candi_itms, candi_lbs):
                cate = str(itm[cate_idx])
                candi_seq[cate_idx_map[cate]].append(itm)
                candi_lb_seq[cate_idx_map[cate]].append(lb)
            for i in range(cate_num):
                seq_len = min(len(candi_seq[i]), candi_max_len)
                candi_len_seq.append(seq_len)
                candi_seq[i] = candi_seq[i][:seq_len] + [[0 for _ in range(len(candi_itms[0]))] for _ in range(candi_max_len - seq_len)]
                candi_lb_seq[i] = candi_lb_seq[i][:seq_len] + [0 for _ in range(candi_max_len - seq_len)]
            proc_candi_itms.append(candi_seq)
            proc_lbs.append(candi_lb_seq)
            proc_candi_lens.append(candi_len_seq)
    else:
        for candi_itms, candi_lbs in zip(candi_lists, candi_lb_lists):
            candi_seq_len = len(candi_itms)
            candi_seq = list(candi_itms[:candi_max_len]) + [[0 for _ in range(len(candi_itms[0]))] for _ in range(candi_max_len - candi_seq_len)]
            candi_lb_seq = list(candi_lbs[:candi_max_len]) + [0 for _ in range(candi_max_len - candi_seq_len)]
            proc_candi_itms.append(candi_seq)
            proc_lbs.append(candi_lb_seq)
            proc_candi_lens.append(min(candi_seq_len, candi_max_len))
        print('proc candi itms', np.array(proc_candi_itms).shape)
        print('proc lb', np.array(proc_lbs).shape)

    split_hist_itms, split_hist_lens = [], []
    union_hist_itms, union_hist_lens = [], []
    if split_hist:
        for hist_itms, hist_len in zip(hist_lists, hist_len_lists):
            hist_cate_seq, hist_cate_len = [[] for _ in range(cate_num)], []
            hist_seq = hist_itms[:hist_len]
            if reverse_hist:
                hist_seq.reverse()
            for itm in hist_seq:
                cate = str(itm[cate_idx])
                hist_cate_seq[cate_idx_map[cate]].append(itm)
            for i in range(cate_num):
                seq_len = len(hist_cate_seq[i])
                hist_cate_len.append(min(seq_len, cate_hist_len))
                if seq_len >= cate_hist_len:
                    if reverse_hist:
                        hist_cate_seq[i] = hist_cate_seq[i][:cate_hist_len]
                    else:
                        hist_cate_seq[i] = hist_cate_seq[i][-cate_hist_len:]
                else:
                    hist_cate_seq[i] = hist_cate_seq[i] + [[0 for _ in range(len(hist_itms[0]))] for _ in range(cate_hist_len - seq_len)]
            split_hist_itms.append(hist_cate_seq)
            split_hist_lens.append(hist_cate_len)
    if union_hist:
        for hist_itms, hist_len in zip(hist_lists, hist_len_lists):
            hist_seq = hist_itms[:hist_len]
            if cloud_hist:
                cur_hist_len = max(0, hist_len - 3)
                hist_seq = hist_seq[:-3]
            else:
                cur_hist_len = hist_len
            if reverse_hist:
                hist_seq.reverse()
            if cur_hist_len >= union_hist_len:
                if reverse_hist:
                    hist_seq = hist_seq[:union_hist_len]
                else:
                    hist_seq = hist_seq[-union_hist_len:]
                cur_hist_len = union_hist_len
            else:
                hist_seq = hist_seq + [[0 for _ in range(len(hist_itms[0]))] for _ in range(union_hist_len - cur_hist_len)]
            union_hist_itms.append(hist_seq)
            union_hist_lens.append(cur_hist_len)

    return usr_ft_lists, proc_candi_itms, proc_candi_lens, proc_lbs, split_hist_itms, split_hist_lens, \
           union_hist_itms, union_hist_lens, full_candi_itms


def evaluate(preds, labels, cate_num, list_len, scopes, exp_exposure_ratio, exposure_scope, is_rank, _print=False):
    ndcg, map = [[] for _ in range(len(scopes))], [[] for _ in range(len(scopes))]
    utility, real_expo_num = [[] for _ in range(len(scopes))], defaultdict(int)
    cates = [[i for _ in range(list_len)] for i in range(cate_num)]
    init_cates = np.array(cates).reshape(-1)
    total_click = []
    for label, pred in zip(labels, preds):
        if is_rank:
            # rerank list
            final = sorted(range(len(pred)), key=lambda k: pred[k], reverse=True)
        else:
            final = list(range(len(pred)))  # evaluate initial rankers

        click = np.array(label)[final].tolist()  # reranked labels
        cates = init_cates[final].tolist()[:exposure_scope]  # reranked cates
        for cate in cates:
            real_expo_num[cate] += 1
        # gold = sorted(range(len(click)), key=lambda k: click[k], reverse=True)  # optimal list for ndcg
        gold = sorted(range(len(click)), key=lambda k: label[k], reverse=True)  # optimal list for ndcg
        for i, scope in enumerate(scopes):
            ideal_dcg, dcg, de_dcg, de_idcg, AP_value, AP_count, util = 0, 0, 0, 0, 0, 0, 0
            cur_scope = min(scope, len(label))
            for _i, _g, _f in zip(range(1, cur_scope + 1), gold[:cur_scope], final[:cur_scope]):
                dcg += (pow(2, click[_i - 1]) - 1) / (np.log2(_i + 1))
                ideal_dcg += (pow(2, label[_g]) - 1) / (np.log2(_i + 1))

                if click[_i - 1] >= 1:
                    AP_count += 1
                    AP_value += AP_count / _i

            _ndcg = float(dcg) / ideal_dcg if ideal_dcg != 0 else 0.
            _map = float(AP_value) / AP_count if AP_count != 0 else 0.
            _click = sum(click[:cur_scope])

            ndcg[i].append(_ndcg)
            map[i].append(_map)
            utility[i].append(_click)

    total_exposure = sum(real_expo_num.values())
    real_expo_ratio = [0 for _ in range(cate_num)]
    for cate, num in real_expo_num.items():
        real_expo_ratio[cate] = num / total_exposure
    JS_diverg = JS_divergence(exp_exposure_ratio, real_expo_ratio)
    return JS_diverg, np.mean(np.array(map), axis=-1).tolist(), np.mean(np.array(ndcg), axis=-1).tolist(), \
           np.mean(np.array(utility), axis=-1).tolist(), [map, ndcg, utility]



