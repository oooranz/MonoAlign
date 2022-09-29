import matplotlib.pyplot as plt
import numpy as np
import palettable
import os
import codecs
import argparse
import model
from utils import get_logger
from fuzzywuzzy import fuzz
import csv
import pandas as pd
from collections import Counter
from matplotlib.ticker import MultipleLocator

LOG = get_logger(__name__)



# datasets = ['MTReference', 'Wikipedia', 'Newsela', 'arXiv']
# l1 = [62.35, 96.29, 73.96, 96.78]
# l2 = [27.79, 2.12, 15.38, 2.26]
# l3 = [9.86, 1.59, 10.66, 0.96]
#
# x = np.arange(4)
#
# total_width, n = 0.8, 3
# width = total_width / n
# x = x - (total_width - width) / 2
#
# plt.figure(figsize=(10, 7), dpi=100)
# plt.rc('font', family='Avenir')
#
# plt.bar(x, l1,  width=width, label='1-1', color=palettable.cartocolors.qualitative.Bold_9.mpl_colors[0])
# plt.bar(x + width, l2, width=width, label='1-n', color=palettable.cartocolors.qualitative.Bold_9.mpl_colors[1])
# plt.bar(x + 2 * width, l3, width=width, label='n-m', color=palettable.cartocolors.qualitative.Bold_9.mpl_colors[2])
# plt.xticks(x+0.8/3,datasets)
# plt.ylabel('Proportion (%)', fontsize=12)
# plt.xlabel('Dataset', fontsize=12)
# plt.legend()
# plt.show()

def jacana_eval(aligns, gs):
    precision_all = []
    recall_all = []
    acc = []
    for test_i in range(len(aligns)):
        pred = set(aligns[test_i].split())
        gold = set(gs[test_i].split())
        if len(pred) > 0:
            precision = len(gold & pred) / len(pred)
        else:
            if len(gold) == 0:
                precision = 1
            else:
                precision = 0

        if len(gold) > 0:
            recall = len(gold & pred) / len(gold)
        else:
            if len(pred) == 0:
                recall = 1
            else:
                recall = 0
        precision_all.append(precision)
        recall_all.append(recall)
        if len(pred & gold) == len(gold) and len(pred & gold) == len(pred):
            acc.append(1)
        else:
            acc.append(0)

    precision = sum(precision_all) / len(precision_all)
    recall = sum(recall_all) / len(recall_all)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    aer = 1 - (sum(precision_all) + sum(recall_all)) / (len(precision_all) + len(recall_all))
    em = sum(acc) / len(acc)
    return precision*100, recall*100, f1*100, aer*100, em*100

def read_mono_dataset(file_name, sure_and_possible=False):
    LOG.info('Reading %s' % file_name)
    data = []
    f = codecs.open(file_name, 'r', 'utf-8')
    for line in f:
        ID, sent1, _, sent2, _, _, _, sure_align, poss_align, *useless = line.strip('\n').split('\t')
        my_dict = {'source': sent1, 'target': sent2, 'sureAlign': sure_align, 'possibleAlign': poss_align, 'id': ID}
        data.append(my_dict)
    source_sentences = []
    target_sentences = []
    alignment_list = []
    idx = []
    for i in range(len(data)):
        source = data[i]['source']
        target = data[i]['target']
        alignment = data[i]['sureAlign']
        index = int(data[i]['id'].split(':')[0])
        if sure_and_possible:
            alignment += ' ' + data[i]['possibleAlign']
        my_label = []
        for item in alignment.split():
            i, j = item.split('-')
            my_label.append(str(i) + '-' + str(j))
        alignment = ' '.join(my_label)

        source_sentences.append(source)
        target_sentences.append(target)
        alignment_list.append(alignment)
        idx.append(index)
    return source_sentences, target_sentences, alignment_list, idx

# test_set = read_mono_dataset('mtref.tsv', True)
test_set = read_mono_dataset('MultiMWA-data/MultiMWA-Wiki/wiki-test.tsv', True)
# test_set = read_mono_dataset('MultiMWA-data/MultiMWA-Newsela/newsela-test.tsv', True)
# test_set = read_mono_dataset('MultiMWA-data/MultiMWA-arXiv/arxiv-test.tsv', True)
f = []
fi = []
fc = []

source_sentences = test_set[0]
target_sentences = test_set[1]
gs = test_set[2]
index = test_set[3]

sim = []
for i in range(len(source_sentences)):
    sim.append(fuzz.token_sort_ratio(source_sentences[i], target_sentences[i]))

print(len(sim))
print(np.mean(sim))
print(np.median(sim))
print(max(sim), min(sim))
# aligner = model.Simalign(matching_methods='a')
# aligns = aligner.align_sentences(source_sentences, target_sentences)
# for n in range(800):
#     f1 = jacana_eval(aligns[:n+1], gs[:n+1])[2]
#     f.append(f1)
#     # print(f1, len(gs))
#
# aligner = model.Simalign(matching_methods='i')
# aligns = aligner.align_sentences(source_sentences, target_sentences)
# for n in range(800):
#     f1 = jacana_eval(aligns[:n+1], gs[:n+1])[2]
#     fi.append(f1)
#
# aligns_o = []
# file = open('jacana-result.txt', 'r')
# for line in file.readlines():
#     line = line.strip()
#     aligns_o.append(line)
# file.close()
# aligns = []
# for i in index:
#     aligns.append(aligns_o[i])
#
# for n in range(800):
#     f1 = jacana_eval(aligns[:n+1], gs[:n+1])[2]
#     fc.append(f1)
#
# plt.figure(figsize=(10,5), dpi=150)
# plt.rc('font', family='Avenir')
#
# plt.plot(range(800),f, color=palettable.cartocolors.qualitative.Bold_9.mpl_colors[0], label='SimAlign-Argmax', linewidth=3)
# plt.plot(range(800),fi, color=palettable.cartocolors.qualitative.Bold_9.mpl_colors[1], label='SimAlign-Itermax', linewidth=3)
# plt.plot(range(800),fc, color=palettable.cartocolors.qualitative.Bold_9.mpl_colors[2], label='Neural semi-CRF Aligner', linewidth=3)
# plt.xlabel('Sample', fontsize=12)
# plt.ylabel('F1 (%)', fontsize=12)
#
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
# plt.tight_layout()
#
# plt.show()

# columns = ['ID', 'sent1', '_1', 'sent2', '_2', '_3', '_4', 'sure_align', 'poss_align', 'useless1', 'sim']
# sid, sent1, sent2, align, sim = [], [], [], [], []
# for i in range(len(source_sentences)):
#     sid.append(str(i) + ':' + str(i))
#     sent1.append(source_sentences[i])
#     sent2.append(target_sentences[i])
#     align.append(gs[i])
#     sim.append(fuzz.token_sort_ratio(source_sentences[i], target_sentences[i]))
#
# new_data = dict(zip(columns, [sid, sent1, 'N/A', sent2, 'N/A', 1, 1, align, None, None, sim]))
# df = pd.DataFrame(new_data)
# # print(df['sim'].median())
# # print(df)
# df.sort_values(["sim"],
#                 axis=0,
#                 ascending=[False],
#                 inplace=True)
# print(df)
# print(df.shape)
# df.to_csv('mtref.tsv', index=False, header=False, sep='\t')


