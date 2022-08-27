import os
import codecs
import argparse
import model
from utils import get_logger
from fuzzywuzzy import fuzz
import csv
import pandas as pd
from collections import Counter

LOG = get_logger(__name__)


def read_mono_dataset(file_name, sure_and_possible=False):
    LOG.info('Reading %s' % file_name)
    data = []
    f = codecs.open(file_name, 'r', 'utf-8')
    for line in f:
        ID, sent1, _, sent2, _, _, _, sure_align, poss_align, *useless = line.strip('\n').split('\t')
        my_dict = {'source': sent1, 'target': sent2, 'sureAlign': sure_align, 'possibleAlign': poss_align}
        data.append(my_dict)
    source_sentences = []
    target_sentences = []
    alignment_list = []
    for i in range(len(data)):
        source = data[i]['source']
        target = data[i]['target']
        alignment = data[i]['sureAlign']
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
    return source_sentences, target_sentences, alignment_list


def align_mono_file(data_set, model_name):
    # Model
    if model_name == 'simalign-argmax':
        aligner = model.Simalign(matching_methods='a')
    elif model_name == 'simalign-itermax':
        aligner = model.Simalign(matching_methods='i')
    elif model_name == 'spanbert-greedy':
        aligner = model.Simalign(model='SpanBERT/spanbert-base-cased', matching_methods='a')
    elif model_name == 'mbert-greedy':
        aligner = model.Simalign(matching_methods='a')

    # Dataset
    if data_set == 'MTReference':
        test_set = read_mono_dataset('MultiMWA-data/MultiMWA-MTRef/mtref-test.tsv', True)
        # test_set = read_mono_dataset('mtref-easy.tsv', True)
    elif data_set == 'Wikipedia':
        test_set = read_mono_dataset('MultiMWA-data/MultiMWA-Wiki/wiki-test.tsv', True)
    elif data_set == 'Newsela':
        test_set = read_mono_dataset('MultiMWA-data/MultiMWA-Newsela/newsela-test.tsv', True)
    elif data_set == 'arXiv':
        test_set = read_mono_dataset('MultiMWA-data/MultiMWA-arXiv/arxiv-test.tsv', True)

    # Read data
    source_sentences = test_set[0]
    target_sentences = test_set[1]
    gs = test_set[2]

    # Word Alignment
    aligns = aligner.align_sentences(source_sentences, target_sentences)

    # Evaluation
    pred_num, gold_num, correct_num_p, correct_num_r, em_num = 0, 0, 0, 0, []
    for align, g in zip(aligns, gs):
        g_items = g.strip().split()
        gold_num += len([edge for edge in g_items if '-' in edge])
        if set(g_items) == set(align.strip().split()):
            em_num.append(1)
        else:
            em_num.append(0)

        for edge in align.strip().split():
            pred_num += 1
            element1 = int(edge.split('-')[0])
            element2 = int(edge.split('-')[1])
            if edge in g_items or '%sp%s' % (element1, element2) in g_items:
                correct_num_p += 1
            if edge in g_items:
                correct_num_r += 1

    prec = correct_num_p / pred_num
    rec = correct_num_r / gold_num
    f1 = 2 * prec * rec / (prec + rec)
    aer = 1 - (correct_num_p + correct_num_r) / (pred_num + gold_num)
    em = sum(em_num) / len(em_num)
    LOG.info('Prec %.3f; Rec %.3f; F1 %.3f; AER %.3f; EM %.3f.' % (prec, rec, f1, aer, em))

def write_to_tsv():
    test_set = read_mono_dataset('MultiMWA-data/MultiMWA-MTRef/mtref-test.tsv', True)
    # test_set = read_mono_dataset('MultiMWA-data/MultiMWA-Wiki/wiki-test.tsv', True)
    # test_set = read_mono_dataset('MultiMWA-data/MultiMWA-Newsela/newsela-test.tsv', True)
    # test_set = read_mono_dataset('MultiMWA-data/MultiMWA-arXiv/arxiv-test.tsv', True)

    source_sentences = test_set[0]
    target_sentences = test_set[1]
    gs = test_set[2]

    columns = ['ID', 'sent1', '_1', 'sent2', '_2', '_3', '_4', 'sure_align', 'poss_align', 'useless1', 'sim']
    sid, sent1, sent2, align, sim = [], [], [], [], []
    for i in range(len(source_sentences)):
        if fuzz.token_sort_ratio(source_sentences[i], target_sentences[i]) >= 77.:
            sid.append(str(i) + ':' + str(i))
            sent1.append(source_sentences[i])
            sent2.append(target_sentences[i])
            align.append(gs[i])
            sim.append(fuzz.token_sort_ratio(source_sentences[i], target_sentences[i]))

    new_data = dict(zip(columns, [sid, sent1, 'N/A', sent2, 'N/A', 1, 1, align, None, None, sim]))
    df = pd.DataFrame(new_data)
    # print(df['sim'].median())
    print(df)
    df.to_csv('mtref-easy.tsv', index=False, header=False, sep='\t')

def count_n_m():
    test_set = read_mono_dataset('MultiMWA-data/MultiMWA-MTRef/mtref-test.tsv', True)
    # test_set = read_mono_dataset('MultiMWA-data/MultiMWA-Wiki/wiki-test.tsv', True)
    # test_set = read_mono_dataset('MultiMWA-data/MultiMWA-Newsela/newsela-test.tsv', True)
    # test_set = read_mono_dataset('MultiMWA-data/MultiMWA-arXiv/arxiv-test.tsv', True)
    gs = test_set[2]
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
            elif (i < len(left)-1 and (left[i] == left[i+1] and right[i]+1 == right[i+1])) or (i > 0 and (left[i] == left[i-1] and right[i] == right[i-1]+1)):
                num_counter.append('n-m')
            else:
                num_counter.append('1-n')

    c = Counter(num_counter)
    print(c)
    total = sum(c.values())
    print('1-1: ', c['1-1'] / total)
    print('1-n: ', c['1-n'] / total)
    print('n-m: ', c['n-m'] / total)

def to_txt():
    # if data_set == 'MTReference':
    #     test_set = read_mono_dataset('MultiMWA-data/MultiMWA-MTRef/mtref-test.tsv', True, False)
    # elif data_set == 'Wikipedia':
    #     test_set = read_mono_dataset('MultiMWA-data/MultiMWA-Wiki/wiki-test.tsv', True, False)
    # elif data_set == 'Newsela':
    #     test_set = read_mono_dataset('MultiMWA-data/MultiMWA-Newsela/newsela-test.tsv', True, False)
    # elif data_set == 'arXiv':
    #     test_set = read_mono_dataset('MultiMWA-data/MultiMWA-arXiv/arxiv-test.tsv', True, False)
    test_set = read_mono_dataset('hard-easy-analysis/mtref-easy.tsv', True)
    # Read data
    source_sentences = test_set[0]
    target_sentences = test_set[1]
    gs = test_set[2]
    aligner = model.Simalign(matching_methods='a')
    aligns = aligner.align_sentences(source_sentences, target_sentences)

    alignments = []
    for sent_id in range(len(source_sentences)):
        alignments.append(source_sentences[sent_id])
        alignments.append(target_sentences[sent_id])
        edges = sorted(gs[sent_id].split(), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))
        for edge in edges:
            if edge not in aligns[sent_id].split():
                e1 = int(edge.split('-')[0])
                e2 = int(edge.split('-')[1])
                word1 = source_sentences[sent_id].split()[e1]
                word2 = target_sentences[sent_id].split()[e2]
                w_pair = word1 + ' - ' + word2
                alignments.append(edge + ' ' + w_pair)
        alignments.append('')

    f = open('hard-easy-analysis/mtref-easy-argmax.txt', 'w')
    for alignment in alignments:
        f.writelines(alignment + '\n')
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-model', type=str, required=True)
    args = parser.parse_args()

    align_mono_file(args.dataset, args.model)

    # write_to_tsv()
    # align_mono_file('MTReference', 'simalign-argmax')
    # to_txt()

    # count_n_m()

