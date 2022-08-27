import model
import itertools
import re
import collections
import numpy as np
import tqdm


def get_span_index(source_sentences, target_sentences):
    src_spans, tgt_spans = [], []
    for sent_id in range(len(source_sentences)):
        src_sent_idx = list(range(len(source_sentences[sent_id].split())))
        tgt_sent_idx = list(range(len(target_sentences[sent_id].split())))
        src_span_idx, tgt_span_idx = [], []
        for d in range(1, 4):
            src_span_idx.extend(
                [src_sent_idx[i: i + d] for i in range(0, len(src_sent_idx)) if i + d <= len(src_sent_idx)])
            tgt_span_idx.extend(
                [tgt_sent_idx[i: i + d] for i in range(0, len(tgt_sent_idx)) if i + d <= len(tgt_sent_idx)])
        src_spans.append(src_span_idx)
        tgt_spans.append(tgt_span_idx)
    return src_spans, tgt_spans


def get_bpe_index(bpe_map, src_idx, tgt_idx, reverse=False):
    spans_pair = []
    for sent_id in range(len(bpe_map)):
        src_spans, tgt_spans = [], []
        if not reverse:
            for src in src_idx[sent_id]:
                if (src[-1] + 1) in bpe_map[sent_id][0]:
                    src_spans.append(
                        list(range(bpe_map[sent_id][0].index(src[0]), bpe_map[sent_id][0].index(src[-1] + 1))))
                else:
                    src_spans.append(list(range(bpe_map[sent_id][0].index(src[0]), len(bpe_map[sent_id][0]))))
            for tgt in tgt_idx[sent_id]:
                if (tgt[-1] + 1) in bpe_map[sent_id][1]:
                    tgt_spans.append(
                        list(range(bpe_map[sent_id][1].index(tgt[0]), bpe_map[sent_id][1].index(tgt[-1] + 1))))
                else:
                    tgt_spans.append(list(range(bpe_map[sent_id][1].index(tgt[0]), len(bpe_map[sent_id][1]))))
        else:
            for src in src_idx[sent_id]:
                if (src[-1] + 1) in bpe_map[sent_id][1]:
                    src_spans.append(
                        list(range(bpe_map[sent_id][1].index(src[0]), bpe_map[sent_id][1].index(src[-1] + 1))))
                else:
                    src_spans.append(list(range(bpe_map[sent_id][1].index(src[0]), len(bpe_map[sent_id][1]))))
            for tgt in tgt_idx[sent_id]:
                if (tgt[-1] + 1) in bpe_map[sent_id][0]:
                    tgt_spans.append(
                        list(range(bpe_map[sent_id][0].index(tgt[0]), bpe_map[sent_id][0].index(tgt[-1] + 1))))
                else:
                    tgt_spans.append(list(range(bpe_map[sent_id][0].index(tgt[0]), len(bpe_map[sent_id][0]))))
        spans_pair.append([src_spans, tgt_spans])
    return spans_pair


def averange(word_tokens_pair):
    w2b_map = []
    cnt = 0
    w2b_map.append([])
    for wlist in word_tokens_pair[0]:
        w2b_map[0].append([])
        for _ in wlist:
            w2b_map[0][-1].append(cnt)
            cnt += 1
    cnt = 0
    w2b_map.append([])
    for wlist in word_tokens_pair[1]:
        w2b_map[1].append([])
        for _ in wlist:
            w2b_map[1][-1].append(cnt)
            cnt += 1
    print(w2b_map)


# def get_alignmatrix_avg(sim_matrix, src_spans, tgt_spans):


# bpemap = [[[0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 8, 8, 8, 9, 10, 10, 10, 10, 11, 12, 12, 12, 13, 14, 15, 15, 15, 15, 16], [0, 1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 8, 8, 8, 9, 10, 10, 10, 10, 11, 12, 12, 12, 13, 14, 15, 15, 15, 15, 16]]]
# source_spans, target_spans = get_span_index(['i love you my shit girl'], ['i hate you'])
# bpemap = [[[0,1,1,2,3,4,5],[0,1,2]]]
# # print(get_span_index(['i love you my shit g s'], ['i hate you']))
#
# # a = [[['In'], ['particular'], [','], ['modern'], ['imaging'], ['and'], ['machine'], ['learning'], ['methods'], ['re', '##ly'], ['strongly'], ['on'], ['maximum', '-', 'a', '-', 'posterior', '##i'], ['estima', '##tion'], ['[', 'E', '##Q', '##U', '##AT', '##ION', ']'], ['whose'], ['calculation'], ['is'], ['a'], ['con', '##ve', '##x'], ['problem'], ['that'], ['can'], ['often'], ['be'], ['solve', '##d'], ['very'], ['efficient', '##ly'], [','], ['even'], ['in'], ['very'], ['high'], ['dimensions'], ['('], ['e', '.', 'g', '.'], [','], ['[', 'MA', '##TH', ']'], [')'], [','], ['by'], ['using'], ['con', '##ve', '##x'], ['op', '##timi', '##sation'], ['techniques'], ['[', 'C', '##IT', '##AT', '##ION', ']'], ['.']], [['In'], ['particular'], [','], ['modern'], ['imaging'], ['and'], ['machine'], ['learning'], ['methods'], ['re', '##ly'], ['strongly'], ['on'], ['maximum', '-', 'a', '-', 'posterior', '##i'], ['('], ['MA', '##P'], [')'], ['estima', '##tion'], ['[', 'E', '##Q', '##U', '##AT', '##ION', ']'], ['whose'], ['calculation'], ['is'], ['a'], ['con', '##ve', '##x'], ['problem'], ['that'], ['can'], ['often'], ['be'], ['solve', '##d'], ['very'], ['efficient', '##ly'], [','], ['even'], ['in'], ['very'], ['high'], ['dimensions'], ['('], ['e', '.', 'g', '.'], [','], ['[', 'MA', '##TH', ']'], [')'], [','], ['by'], ['using'], ['con', '##ve', '##x'], ['op', '##timi', '##sation'], ['techniques'], ['[', 'C', '##IT', '##AT', '##ION', ']'], ['.']]]
# # print(a)
# print(source_spans)
# source_words, target_words = [], []
# for src_sent, tgt_sent in zip(source_spans, target_spans):
#     src_words, tgt_words = [], []
#     src_words.extend([span for span in src_sent if len(span) == 1])
#     tgt_words.extend([span for span in tgt_sent if len(span) == 1])
#     source_words.append(src_words)
#     target_words.append(tgt_words)
# # # source_words = [span for sent in source_seg_idx for span in sent if len(span) == 1]
# # print(source_words)
# s2t_spans_pair_bpe = get_bpe_index(bpemap, source_spans, target_words)
# t2s_spans_pair_bpe = get_bpe_index(bpemap, target_spans, source_words, True)
# print(s2t_spans_pair_bpe)
# print(t2s_spans_pair_bpe)
# averange(span[0])
# ss = ['as the curtain of darkness arrived , troops tried to disperse those young men throwing stones at soldiers in the chinatown areas by firing their guns as warning .']
# ts = ['with the approach of nightfall , the army fired shots to disperse groups of young people who were throwing rocks at the soldiers .']
ss = ['Wall street stocks fell sharply .']
ts = ['Stocks slump on wall street .']
# ss = ["japan keeps its status as the biggest trade partner , and the us and hongkong are in the first and second place respectively ."]
# ts = ["japan is the biggest trade partner followed by the united state of america and hong kong ."]


def split_list_by_n(list_collection, n):
    for i in range(0, len(list_collection), n):
        yield list_collection[i: i + n]


def split_list_by_n_add(list_collection, n, last_len):
    for i in range(last_len, len(list_collection), n):
        yield list_collection[i: i + n]


# ss = ['i love u']
# ts = ['o h a']
# s_spans, t_spans = get_span_index(ss, ts)
# print(s_spans)
# print(t_spans)
a = [[1, 2], [3]]
b = [[0, 1], [2]]


# s_len, t_len = len(ss[0].split()), len(ts[0].split())
# # # lenth = 2
# a = [list(split_list_by_n_add(list(range(s_len)), 2, 0))]
# a = list(split_list_by_n_add(list(range(s_len)), 2, 0))
# print(a)

def get_combinations(source_len, target_len):
    def split_list_by_n(list_collection, n, last_len=0):
        for i in range(last_len, len(list_collection), n):
            yield list_collection[i: i + n]

    def get_combination(list1, list2):
        if len(list1) < len(list2):
            combination = [list(zip(list1, p)) for p in itertools.permutations(list2)]
        else:
            combination = [list(zip(p, list2)) for p in itertools.permutations(list1)]
        return combination

    combinations = []
    # length = 1
    combinations.extend(get_combination([[i] for i in range(source_len)], [[i] for i in range(target_len)]))

    # length = 2
    src_two = [list(split_list_by_n(list(range(source_len)), 2))]
    tgt_two = [list(split_list_by_n(list(range(target_len)), 2))]
    if len(src_two[0][-1]) == 1:
        src_two += [[[0]] + list(split_list_by_n(list(range(source_len)), 2, 1))]
    if len(tgt_two[0][-1]) == 1:
        tgt_two += [[[0]] + list(split_list_by_n(list(range(target_len)), 2, 1))]
    prod_two = list(itertools.product(src_two, tgt_two))
    for prod in prod_two:
        combinations.extend(get_combination(prod[0], prod[1]))

    # length = 3
    src_three = [list(split_list_by_n(list(range(source_len)), 3))]
    tgt_three = [list(split_list_by_n(list(range(target_len)), 3))]
    if len(src_three[0][-1]) == 1:
        src_three += [[[0]] + list(split_list_by_n(list(range(source_len)), 3, 1))]
    if len(tgt_three[0][-1]) == 1:
        tgt_three += [[[0]] + list(split_list_by_n(list(range(target_len)), 3, 1))]
    if len(src_three[0][-1]) == 2:
        src_three += [[[0]] + list(split_list_by_n(list(range(source_len)), 3, 2))]
    if len(tgt_three[0][-1]) == 2:
        tgt_three += [[[0]] + list(split_list_by_n(list(range(target_len)), 3, 2))]
    prod_three = list(itertools.product(src_three, tgt_three))
    for prod in prod_three:
        combinations.extend(get_combination(prod[0], prod[1]))

    return combinations


# combinations = get_combinations(3, 3)


def get_max_combination(combs, sim_df):
    comb_scores = collections.defaultdict(lambda: [])
    for comb in combs:
        sims = []
        for span_align in comb:
            sim = sim_df.at[str(span_align[0]), str(span_align[1])]
            if sim > 0:
                sims.append(sim)
        comb_scores[comb].append(np.mean(sims))
    return sorted(comb_scores, key=lambda x: comb_scores[x])[-1]


# print(get_combinations(3, 3))

# last = len(a[-1])
# if last < 2:


# b = a.copy()
# c = [a[i] + b[i + 1] for i in range(len(a) - 1)]
# print(c)
# b=c
# combination = []
# if len(a) < len(b):
#     combination += [list(zip(a, p)) for p in itertools.permutations(b)]
# else:
#     combination += [list(zip(p, b)) for p in itertools.permutations(a)]
# # combination = [[[x,y] for x in a] for y in b]
# print(combination)

# ss = ['i love you']
# ts = ['i you']
# ss = ['after the signing ceremony , president akaev and premier li peng answered the questions from reporters respectively .']
# ts = ['after the signing ceremonym president akaev and premier li peng answered questions of reporters .']
# ss = ['the success story of ding hao is a result of joint efforts from various circles of the society .']
# ts = ['ding owes his success to people from all walks of life in society .']
# id_word_aligner = model.Simalign(matching_methods='a')
# aligns = id_word_aligner.align_spans_iter(ss, ts)
# # aligns = id_word_aligner.align_sentences(ss, ts)
# print(aligns)
# aligns2 = id_word_aligner.align_sentences(ss, ts)
# aligner = model.Simalign(model='SpanBERT/spanbert-base-cased', matching_methods='a')
# print(aligner.align_spans(ss, ts))
# aligner = model.Simalign(matching_methods='i')
# aligns3 = aligner.align_sentences(ss, ts)
# print(aligns)
# print(aligns2)
# print('2-2 3-3 6-5 13-11 17-14 5-4 10-9 7-6 11-10 0-0 9-8 14-12 8-7 1-1 15-13 12-11')

# def comb(rows):
#     combs = []
#     def next_c(li = 0):
#         for lj in xrange(li, len(rows)):
#             one[]

#
# print(get_combination([[i] for i in range(14)], [[i] for i in range(17)]))
# lista = [[i] for i in range(14)]
# print(itertools.permutations(lista))

# src_spans, tgt_spans = [[0], [2], [3], [2, 3], [3, 4], [2, 3, 4]], [[0], [1], [3], [0, 1], [1, 2], [2, 3], [0, 1, 2],
#                                                                     [1, 2, 3]]


def get_combs(src_spans, tgt_spans, df):
    def get_span_dict(spans):
        span_dict = {}
        for span in spans:
            if len(span) == 1:
                span_dict[span[0]] = list(filter(lambda x: span[0] in x, spans))
        return span_dict

    def cartesian_iterative(pools):
        result = [[]]
        for pool in pools:
            if result == [[]]:
                result = [x + [y] for x in result for y in pool]
            else:
                result = [x + [y] for x in result for y in pool if not set(y) & set(x[-1]) or y == x[-1]]
        return [np.unique(i).tolist() for i in np.array(result, dtype=object)]

    def get_combination(list1, list2):
        return [zip(list1, p) for p in itertools.permutations(list2)]

    prod_src, prod_tgt = cartesian_iterative(get_span_dict(src_spans).values()), cartesian_iterative(get_span_dict(tgt_spans).values())
    # print(prod_src)
    # print(prod_tgt)
    combinations = []
    for src in prod_src:
        for tgt in prod_tgt:
            if len(src) == len(tgt):
                combinations.extend(get_combination(src, tgt))
    # Get the combination with the largest average similarity
    comb_scores = collections.defaultdict(lambda: [])
    for comb in combinations:
        sims = []
        for align in comb:
            sims.append(df.at[str(align[0]), str(align[1])])
        comb_scores[comb].append(np.mean(sims))

    return sorted(comb_scores, key=lambda x: comb_scores[x])[-1]
# get_combs(src_spans, tgt_spans)


def cartesian_iterative(pools):
    result = [[]]
    for pool in pools:
        if result == [[]]:
            result = [x + [y] for x in result for y in pool]
        else:
            result = [x + [y] for x in result for y in pool if not set(y) & set(x[-1]) or y == x[-1]]
    return [np.unique(i).tolist() for i in np.array(result, dtype=object)]
# print(cartesian_iterative([[[0]], [[2], [2, 3], [2, 3, 4]], [[3], [2, 3], [3, 4], [2, 3, 4]]]))
# print(cartesian_iterative([[[0], [0, 1], [0, 1, 2]], [[1], [0, 1], [1, 2], [0, 1, 2], [1, 2, 3]], [[3], [2, 3], [1, 2, 3]]]))

from collections import Counter
gs = ['14-16 9-10 2-4 0-0 17-21 11-12 13-15 28-23 11-13 25-9 12-14 23-9 16-20 6-5 25-8 15-19 7-6 24-8 1-1 5-2 4-4 23-8 24-9 7-7 10-11 14-18 17-22 3-4 14-17']
num_counter = []

for sent in gs:
    sorted_sent = sorted(sent.strip().split(), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))
    left = []
    right = []
    for edge in sorted_sent:
        element1 = int(edge.split('-')[0])
        element2 = int(edge.split('-')[1])
        left.append(element1)
        right.append(element2)
    for i in range(len(left)):
        if Counter(left)[left[i]] == Counter(right)[right[i]] == 1:
            num_counter.append('1-1')
        elif Counter(left)[left[i]] == 1 or Counter(right)[right[i]] == 1:
            num_counter.append('1-n')
        else:
            num_counter.append('n-m')
print(Counter(num_counter))
