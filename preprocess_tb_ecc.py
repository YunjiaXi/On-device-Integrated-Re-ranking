import os
import csv
import random
from collections import defaultdict
import pickle as pkl
# import pathos
import json
import numpy as np

FILTER_POS_NUM = 1
FILTER_NEG_NUM = 1
FILTER_HIST_NUM = 0
CANDIDATE_NEG_NUM = 9
HIST_MAX_LEN = 30
HIST_CATE_MAX_LEN = 20

# user_id,user_os,user_gender,user_age_level,user_purchase_level,user_hour,  # 0-5
# cand_item_pos,cand_item_pagenum,cand_item_sex,cand_item_price_level,cand_item_age_level,cand_item_bc_type, # 6-11
# exp_item_pos_seq,exp_item_pagenum_seq,exp_item_sex_seq,exp_item_price_level_seq,
# exp_item_age_level_seq,exp_item_bc_type_seq, # 12-17
# ipv_item_pos_seq,ipv_item_pagenum_seq,ipv_item_sex_seq,ipv_item_price_level_seq,
# ipv_item_age_level_seq,ipv_item_bc_type_seq, # 18-23
# label,score # 24-25


def load_data(data_path):
    user_ft, pos_item_hist, item_ft = {}, {}, {}
    user_neg_item, user_pos_item = defaultdict(list), defaultdict(list)
    csv_reader = csv.reader(open(data_path, 'r'))
    head = next(csv_reader)
    idx = 0
    # pos_item_num, hist_total_len = 0, 0
    min_score, max_score = 1e9, -1e9
    for line in csv_reader:
        user = line[0]
        if not user_ft.__contains__(user):
            user_ft[user] = line[1:6]
        label = int(line[-2])
        score = line[-1]
        min_score = min(min_score, float(score))
        max_score = max(max_score, float(score))
        item_ft[idx] = line[6:12] + [round(float(score)*10)]
        if label:
            pos_item_hist[idx] = get_hist(line[18:24])
            user_pos_item[user].append(idx)
        else:
            user_neg_item[user].append(idx)
        idx += 1
    print('min score', min_score, 'max score', max_score)
    # statistics
    print('statistics before filter')
    stats(user_ft, user_pos_item, user_neg_item, item_ft, pos_item_hist)

    # filter
    rm_usr = []
    for user in user_ft:
        pos_list = []
        for pos_itm in user_pos_item[user]:
            if len(pos_item_hist[pos_itm]) > FILTER_HIST_NUM:
                pos_list.append(pos_itm)
            else:
                item_ft.pop(pos_itm)
        user_pos_item[user] = pos_list
        pos_num, neg_num = len(pos_list), len(user_neg_item[user])
        if not (pos_num >= FILTER_POS_NUM and neg_num >= FILTER_NEG_NUM):
            rm_usr.append(user)
            for item in user_pos_item[user]:
                item_ft.pop(item)
                pos_item_hist.pop(item)
            for item in user_neg_item[user]:
                item_ft.pop(item)
            user_pos_item.pop(user)
            user_neg_item.pop(user)

    for user in rm_usr:
        user_ft.pop(user)
    print('statistics after filter')
    stats(user_ft, user_pos_item, user_neg_item, item_ft, pos_item_hist)

    return user_ft, user_pos_item, user_neg_item, item_ft, pos_item_hist


def get_hist(hist_seq_ft):
    # (ft_num, seq_len)
    fts_seq = [i.split(',') for i in hist_seq_ft]
    ft_num = len(hist_seq_ft)
    seq_len, max_len = 0, len(fts_seq[0])
    for i in range(max_len):
        t_sum = sum([int(fts_seq[j][i]) for j in range(ft_num)])
        if t_sum == 0:
            if i + 1 < max_len and sum([int(fts_seq[j][i+1]) for j in range(ft_num)]) == 0:
                break
        else:
            seq_len += 1
    new_seq = [fts_seq[i][:seq_len] for i in range(ft_num)]
    return np.array(new_seq).transpose().tolist()  # (seq_len, ft_num)


def stats(user_ft, user_pos_item, user_neg_item, item_ft, pos_item_hist):
    user_num, item_num = len(user_ft), len(item_ft)
    print('user num', user_num, 'pos item num', len(pos_item_hist), 'total item num', item_num, len(pos_item_hist)/item_num)
    user_pos_1, user_pos_2, user_neg_1, user_neg_9, user_pos1_neg2, user_pos1_neg9, user_pos1_neg1, \
                user_pos1_neg5 = 0, 0, 0, 0, 0, 0, 0, 0
    pos_cate_num, hist_cate_num, cate_num = defaultdict(int), defaultdict(int), defaultdict(int)
    total_pos_hist_len, pos_hist_len_le10, pos_hist_len_g10, pos_hist_len_le5, pos_hist_len_e0 = 0, 0, 0, 0, 0

    for user in user_ft:
        pos_num, neg_num = len(user_pos_item[user]), len(user_neg_item[user])
        for pos_itm in user_pos_item[user]:
            hist_len = len(pos_item_hist[pos_itm])
            total_pos_hist_len += hist_len
            if hist_len < 1:
                pos_hist_len_e0 += 1
            elif hist_len <= 5:
                pos_hist_len_le5 += 1
            elif hist_len <= 10:
                pos_hist_len_le10 += 1
            else:
                pos_hist_len_g10 += 1
            itm_cate = item_ft[pos_itm][-2]
            pos_cate_num[itm_cate] += 1
            cate_num[itm_cate] += 1
            for hist_itm in pos_item_hist[pos_itm]:
                hist_cate_num[hist_itm[-1]] += 1

        for neg_itm in user_neg_item[user]:
            itm_cate = item_ft[neg_itm][-2]
            cate_num[itm_cate] += 1

        if pos_num >= 1:
            user_pos_1 += 1
            if pos_num >= 2:
                user_pos_2 += 1
            if neg_num >= 1:
                user_pos1_neg1 += 1
                if neg_num >= 2:
                    user_pos1_neg2 += 1
                    if neg_num >= 5:
                        user_pos1_neg5 += 1
                        if neg_num >= 9:
                            user_pos1_neg9 += 1
        if neg_num >= 1:
            user_neg_1 += 1
            if neg_num >= 9:
                user_neg_9 += 1

    print('pos >= 1:', user_pos_1 / user_num, 'pos >= 2:', user_pos_2 / user_num)
    print('neg >= 1:', user_neg_1 / user_num, 'neg >= 9:', user_neg_9 / user_num)
    print('neg>=1,pos>=1', user_pos1_neg1, user_pos1_neg1/user_num, 'neg>=9,pos>=1', user_pos1_neg9,
          user_pos1_neg9/user_num, 'neg>=5,pos>=1', user_pos1_neg5, user_pos1_neg5/user_num,
          'neg>=2,pos>=1', user_pos1_neg2, user_pos1_neg2/user_num)

    pos_itm_num = len(pos_item_hist)
    print('average hist len:', total_pos_hist_len/pos_itm_num, 'hist==0:', pos_hist_len_e0, pos_hist_len_e0/pos_itm_num,
          '0<hist<=5:', pos_hist_len_le5, pos_hist_len_le5/pos_itm_num, '5<hist<=10:', pos_hist_len_le10,
          pos_hist_len_le10/pos_itm_num, 'hist>10:', pos_hist_len_g10, pos_hist_len_g10/pos_itm_num)

    print('item cate', cate_num.keys(), 'pos item cate', pos_cate_num.keys(), 'hist cate', hist_cate_num.keys())
    print('item cate')
    for key in cate_num:
        print('cate', key, cate_num[key], cate_num[key] / item_num)
    print('pos item cate')
    for key in pos_cate_num:
        print('cate', key, pos_cate_num[key], pos_cate_num[key]/len(pos_item_hist))
    print('hist cate')
    for key in hist_cate_num:
        print('cate', key, hist_cate_num[key], hist_cate_num[key]/total_pos_hist_len)


def generate_dataset(usr_ft, usr_pos_itm, usr_neg_itm, itm_ft, pos_itm_hist, uid_map, usr_ft_map, itm_ft_map):
    # cate_num = len(cate_name_map)
    neg_item_set = []
    for usr in usr_neg_itm:
        neg_item_set.extend(usr_neg_itm[usr])

    def generate_lists_per_user(data):
        user, pos_itms, neg_itms = data
        hist_lists, hist_cate_lists, usr_ft_lists, candi_lists, hist_cate_len_lists, hist_len_lists, \
                    candi_len_lists, candi_lb_lists = [], [], [], [], [], [], [], []
        usr_ft_id = [uid_map[user]] + [usr_ft_map[i][ft] for i, ft in enumerate(usr_ft[user])]
        for pos_itm in pos_itms:
            usr_ft_lists.append(usr_ft_id)
            if len(neg_itms) >= CANDIDATE_NEG_NUM:
                selected_neg_itms = random.sample(neg_itms, CANDIDATE_NEG_NUM)
            else:
                selected_neg_itms = neg_itms + random.sample(neg_item_set, CANDIDATE_NEG_NUM - len(neg_itms))
            candi_itms = [pos_itm] + selected_neg_itms
            candi_lbs = [1] + [0 for _ in range(len(selected_neg_itms))]

            pairs = list(zip(candi_itms, candi_lbs))
            random.shuffle(pairs)
            candi_itms, candi_lbs = list(zip(*pairs))
            hist_itms = pos_itm_hist[pos_itm]
            hist_seq = []
            for itm in hist_itms:
                ft_id = [itm_ft_map[i][ft] for i, ft in enumerate(itm)]
                hist_seq.append(ft_id)
            candi_seq = []
            for itm in candi_itms:
                ft_id = [itm_ft_map[i][ft] for i, ft in enumerate(itm_ft[itm])]
                candi_seq.append(ft_id)

            candi_lists.append(candi_seq)
            candi_lb_lists.append(candi_lbs)
            hist_seq_len = len(hist_seq)
            hist_len_lists.append(min(HIST_MAX_LEN, hist_seq_len))
            if hist_seq_len < HIST_MAX_LEN:
                hist_seq = hist_seq + [[0 for _ in range(6)] for _ in range(HIST_MAX_LEN - hist_seq_len)]
            else:
                hist_seq = hist_seq[:HIST_MAX_LEN]
            hist_lists.append(hist_seq)
        return usr_ft_lists, candi_lists, candi_lb_lists, hist_lists, hist_len_lists

    total_usr_ft_lists, total_candi_lists, total_candi_lb_lists, total_hist_cate_lists, total_hist_lists, \
            total_candi_len_lists, total_hist_cate_len_lists, total_hist_len_lists = [], [], [], [], [], [], [], []
    for usr in usr_pos_itm:
        data = usr, usr_pos_itm[usr], usr_neg_itm[usr]
        usr_ft_lists, candi_lists, candi_lb_lists, hist_lists, hist_len_lists = generate_lists_per_user(data)
        total_usr_ft_lists.extend(usr_ft_lists)
        total_candi_lists.extend(candi_lists)
        total_candi_lb_lists.extend(candi_lb_lists)
        total_hist_lists.extend(hist_lists)
        total_hist_len_lists.extend(hist_len_lists)

    return total_usr_ft_lists, total_candi_lists, total_candi_lb_lists, total_hist_lists, total_hist_len_lists


def process_data(train_path, test_path, processed_path):
    # load and filter data
    print('load train dataset')
    if os.path.exists(processed_path + 'train.filter'):
        train_u_ft, train_u_pos_itm, train_u_neg_itm, train_i_ft, train_pos_i_hist = pkl.load(open(processed_path + 'train.filter', 'rb'))
    else:
        train_u_ft, train_u_pos_itm, train_u_neg_itm, train_i_ft, train_pos_i_hist = load_data(train_path)
        pkl.dump([train_u_ft, train_u_pos_itm, train_u_neg_itm, train_i_ft, train_pos_i_hist], open(processed_path + 'train.filter', 'wb'))
    print('load test dataset')
    if os.path.exists(processed_path + 'test.filter'):
        test_u_ft, test_u_pos_itm, test_u_neg_itm, test_i_ft, test_pos_i_hist = pkl.load(open(processed_path + 'test.filter', 'rb'))
    else:
        test_u_ft, test_u_pos_itm, test_u_neg_itm, test_i_ft, test_pos_i_hist = load_data(test_path)
        pkl.dump([test_u_ft, test_u_pos_itm, test_u_neg_itm, test_i_ft, test_pos_i_hist], open(processed_path + 'test.filter', 'wb'))

    # get feature map
    uid_map, usr_ft_map, itm_ft_map = {}, [{} for _ in range(6)], [{} for _ in range(7)]
    idx = 1
    for itm_ft_data in [train_i_ft, test_i_ft]:
        for itm_fts in itm_ft_data.values():
            for i, ft in enumerate(itm_fts):
                if not itm_ft_map[i].__contains__(ft):
                    itm_ft_map[i][ft] = idx
                    idx += 1
    for hist_ft_data in [train_pos_i_hist, test_pos_i_hist]:
        for hist_seq_fts in hist_ft_data.values():
            for hist_itm in hist_seq_fts:
                for i, ft in enumerate(hist_itm):
                    if not itm_ft_map[i].__contains__(ft):
                        itm_ft_map[i][ft] = idx
                        idx += 1
    for usr_ft_data in [train_u_ft, test_u_ft]:
        for usr, usr_fts in usr_ft_data.items():
            if not uid_map.__contains__(usr):
                uid_map[usr] = idx
                idx += 1
            for i, ft in enumerate(usr_fts):
                if not usr_ft_map[i].__contains__(ft):
                    usr_ft_map[i][ft] = idx
                    idx += 1
    print('total feature num', idx, 'uid num', len(uid_map))
    usr_field_name = ['user_os', 'user_gender', 'user_age_level', 'user_purchase_level', 'user_hour']
    itm_field_name = ['pos', 'pagenum', 'sex', 'price_level', 'age_level', 'bc_type', 'score']
    print('user ft num:')
    for i, j in zip(usr_field_name, usr_ft_map):
        print(i, len(j), end='\t')
    print('\nitem ft num:')
    for i, j in zip(itm_field_name, itm_ft_map):
        print(i, len(j), end='\t')
    print()

    # generate samples
    cate_idx_map = {name: idx for idx, name in enumerate(list(itm_ft_map[-2].values()))}
    cate_name_map = itm_ft_map[-2]
    print('cate idx map: ', cate_idx_map, 'cate name map', cate_name_map)
    train_set = generate_dataset(train_u_ft, train_u_pos_itm, train_u_neg_itm, train_i_ft, train_pos_i_hist,
                     uid_map, usr_ft_map, itm_ft_map)
    test_set = generate_dataset(test_u_ft, test_u_pos_itm, test_u_neg_itm, test_i_ft, test_pos_i_hist,
                     uid_map, usr_ft_map, itm_ft_map)

    test_data, train_data = list(zip(*test_set)), list(zip(*train_set))
    random.shuffle(test_data)
    random.shuffle(train_data)
    test_data_len, train_data_len = len(test_data), len(train_data)
    print('The number of page: train', train_data_len, 'test', test_data_len, 'page per user:',
          (train_data_len+test_data_len) / len(uid_map))

    train_set, test_set = list(zip(*train_data)), list(zip(*test_data))

    print(np.array(train_set[0]).shape, np.array(train_set[1]).shape, np.array(train_set[2]).shape,
          np.array(train_set[3]).shape, np.array(train_set[4]).shape)
    print(np.array(test_set[0]).shape, np.array(test_set[1]).shape, np.array(test_set[2]).shape,
          np.array(test_set[3]).shape, np.array(test_set[4]).shape)

    # save
    stat = {'itm_fnum': 7, 'cate_num': 3, 'list_len': 10, 'hist_len': 30, 'cate_idx_map': cate_idx_map,
            'ft_num': idx,  'user_num': len(uid_map), 'hist_fnum': 6, 'usr_fnum': 6, 'cate_idx': 5,
            'cate_name_map': cate_name_map}
    print('stat', stat)
    with open(processed_path + 'data.stat', 'w') as f:
        stat = json.dumps(stat)
        f.write(stat)
    for postfix, date in [('train', train_set), ('test', test_set)]:
        with open(processed_path + 'data.' + postfix, 'wb') as f:
            pkl.dump(date, f)
    print(' =============data process done=============')


if __name__ == '__main__':
    processed_path = './data/taobao/process/'
    if not os.path.exists(processed_path):
        os.mkdir(processed_path)
    train_path = './data/taobao/raw/Part1-taobao_open_mcc_train.csv'
    test_path = './data/taobao/raw/Part1-taobao_open_mcc_test.csv'
    process_data(train_path, test_path, processed_path)

