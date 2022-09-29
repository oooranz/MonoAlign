import os
import codecs
import argparse
import model
import re
import spacy
import pandas as pd
import numpy as np
from spacy.tokenizer import Tokenizer
from collections import Counter
from utils import get_logger

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
        aligner = model.Simalign(model='spanbert', matching_methods='a')
    elif model_name == 'mbert-greedy':
        aligner = model.Simalign(matching_methods='a')

    # Dataset
    if data_set == 'MTReference':
        test_set = read_mono_dataset('MultiMWA-data/MultiMWA-MTRef/mtref-test.tsv', True)
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
    # aligns = []
    # f = open('jacana-result.txt', 'r')
    # for line in f.readlines():
    #     line = line.strip()
    #     aligns.append(line)
    # f.close()

    # Evaluation
    prec, rec, f1, aer, em = jacana_eval(aligns, gs)
    LOG.info('Prec %.1f; Rec %.1f; F1 %.1f; AER %.1f; EM %.1f.' % (prec, rec, f1, aer, em))

def simalign_eval(aligns, gs):
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

    if pred_num == 0 or gold_num == 0 or correct_num_p == 0 or correct_num_r == 0:
        f1 = 0.
    else:
        prec = correct_num_p / pred_num
        rec = correct_num_r / gold_num
        f1 = 2 * prec * rec / (prec + rec)
        aer = 1 - (correct_num_p + correct_num_r) / (pred_num + gold_num)
        em = sum(em_num) / len(em_num)
    return f1 * 100

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

def pos_num_analysis(data_set):
    # Dataset
    if data_set == 'MTReference':
        test_set = read_mono_dataset('MultiMWA-data/MultiMWA-MTRef/mtref-test.tsv', True)
    elif data_set == 'Wikipedia':
        test_set = read_mono_dataset('MultiMWA-data/MultiMWA-Wiki/wiki-test.tsv', True)
    elif data_set == 'Newsela':
        test_set = read_mono_dataset('MultiMWA-data/MultiMWA-Newsela/newsela-test.tsv', True)
    elif data_set == 'arXiv':
        test_set = read_mono_dataset('MultiMWA-data/MultiMWA-arXiv/arxiv-test.tsv', True)

    # Read data
    source_sentences = test_set[0]
    target_sentences = test_set[1]

    special_cases = {":)": [{"ORTH": ":)"}]}
    prefix_re = re.compile(r'''''')
    suffix_re = re.compile(r'''''')
    infix_re = re.compile(r'''''')
    simple_url_re = re.compile(r'''''')

    def custom_tokenizer(nlp):
        return Tokenizer(nlp.vocab, rules=special_cases,
                         prefix_search=prefix_re.search,
                         suffix_search=suffix_re.search,
                         infix_finditer=infix_re.finditer,
                         url_match=simple_url_re.match)

    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = custom_tokenizer(nlp)

    pos_list = []
    for i in range(len(source_sentences)):
        doc1 = nlp(source_sentences[i])
        doc2 = nlp(target_sentences[i])
        pos1, pos2 = [], []
        for token1 in doc1:
            pos1.append(token1.pos_)
            pos_list.append(token1.pos_)
        for token2 in doc2:
            pos2.append(token2.pos_)
            pos_list.append(token2.pos_)
        if len(pos1) != len(source_sentences[i].split()) or len(pos2) != len(target_sentences[i].split()):
            print('shit')
    new_dict = dict()
    for key, value in Counter(pos_list).items():
        new_dict[key] = '{} | {:.1f}%'.format(value, value * 100 / len(pos_list))
    # return pd.DataFrame.from_dict(Counter(pos_list), orient='index')
    return pd.DataFrame.from_dict(new_dict, orient='index')

def gs_pos_dist(data_set):
    # Dataset
    if data_set == 'MTReference':
        test_set = read_mono_dataset('MultiMWA-data/MultiMWA-MTRef/mtref-test.tsv', True)
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
    gs_num = 0
    for g in gs:
        gs_num += len(g.split())

    special_cases = {":)": [{"ORTH": ":)"}]}
    prefix_re = re.compile(r'''''')
    suffix_re = re.compile(r'''''')
    infix_re = re.compile(r'''''')
    simple_url_re = re.compile(r'''''')

    def custom_tokenizer(nlp):
        return Tokenizer(nlp.vocab, rules=special_cases,
                         prefix_search=prefix_re.search,
                         suffix_search=suffix_re.search,
                         infix_finditer=infix_re.finditer,
                         url_match=simple_url_re.match)

    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = custom_tokenizer(nlp)

    pos_tags = ['ADP', 'DET', 'NOUN', 'PUNCT', 'PROPN', 'CCONJ', 'VERB', 'ADV', 'SCONJ', 'PART', 'ADJ', 'PRON', 'AUX',
                'NUM', 'SYM', 'INTJ', 'X']
    ratio_mat = np.zeros((17, 17))
    num_mat = np.zeros((17, 17))
    for x in range(17):
        for y in range(17):
            # Get pos id
            pos1 = pos_tags[x]
            pos2 = pos_tags[y]
            src_pos_id1, src_pos_id2, tgt_pos_id1, tgt_pos_id2 = [], [], [], []
            for i in range(len(source_sentences)):
                src_doc = nlp(source_sentences[i])
                tgt_doc = nlp(target_sentences[i])
                src_pos1, src_pos2, tgt_pos1, tgt_pos2 = [], [], [], []
                for token_id in range(len(src_doc)):
                    if src_doc[token_id].pos_ == pos1:
                        src_pos1.append(token_id)
                    if src_doc[token_id].pos_ == pos2:
                        src_pos2.append(token_id)
                for token_id in range(len(tgt_doc)):
                    if tgt_doc[token_id].pos_ == pos1:
                        tgt_pos1.append(token_id)
                    if tgt_doc[token_id].pos_ == pos2:
                        tgt_pos2.append(token_id)
                src_pos_id1.append(src_pos1)
                src_pos_id2.append(src_pos2)
                tgt_pos_id1.append(tgt_pos1)
                tgt_pos_id2.append(tgt_pos2)

            # POS Evaluation
            golds = []
            for i in range(len(gs)):
                gold = []
                for edge in gs[i].split():
                    e1 = int(edge.split('-')[0])
                    e2 = int(edge.split('-')[1])
                    if (e1 in src_pos_id1[i] and e2 in tgt_pos_id2[i]) or (
                            e1 in src_pos_id2[i] and e2 in tgt_pos_id1[i]):
                        golds.append(edge)
                golds.append(' '.join(list(set(gold))))
            num = 0
            for sub_g in golds:
                num += len(sub_g.split())
            ratio_mat[x][y] = (num / gs_num) * 100
            num_mat[x][y] = num
            LOG.info('{} - {}: {} / {} = {:.1%}'.format(pos1, pos2, num, gs_num, num / gs_num))
    return pd.DataFrame(ratio_mat), pd.DataFrame(num_mat)

def pos_analysis(data_set, model_name):
    # Dataset
    if data_set == 'MTReference':
        test_set = read_mono_dataset('MultiMWA-data/MultiMWA-MTRef/mtref-test.tsv', True)
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

    # Model
    if model_name == 'simalign-argmax':
        aligner = model.Simalign(matching_methods='a')
        aligns = aligner.align_sentences(source_sentences, target_sentences)
    elif model_name == 'simalign-itermax':
        aligner = model.Simalign(matching_methods='i')
        aligns = aligner.align_sentences(source_sentences, target_sentences)
    elif model_name == 'neural-semi-CRF':
        aligns = []
        f = open('jacana-result.txt', 'r')
        for line in f.readlines():
            line = line.strip()
            aligns.append(line)
        f.close()

    special_cases = {":)": [{"ORTH": ":)"}]}
    prefix_re = re.compile(r'''''')
    suffix_re = re.compile(r'''''')
    infix_re = re.compile(r'''''')
    simple_url_re = re.compile(r'''''')

    def custom_tokenizer(nlp):
        return Tokenizer(nlp.vocab, rules=special_cases,
                         prefix_search=prefix_re.search,
                         suffix_search=suffix_re.search,
                         infix_finditer=infix_re.finditer,
                         url_match=simple_url_re.match)

    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = custom_tokenizer(nlp)

    src_pos_id, tgt_pos_id = [], []
    for i in range(len(source_sentences)):
        src_doc = nlp(source_sentences[i])
        tgt_doc = nlp(target_sentences[i])
        src_pos, tgt_pos = [], []
        for stoken in src_doc:
            src_pos.append(stoken.pos_)
        for ttoken in tgt_doc:
            tgt_pos.append(ttoken.pos_)

        src_pos_id.append(src_pos)
        tgt_pos_id.append(tgt_pos)

    predictions, golds = [], []
    num_gs = 0
    num_same = 0
    for i in range(len(gs)):
        pred, gold = [], []
        for align_edge in aligns[i].split():
            e1 = int(align_edge.split('-')[0])
            e2 = int(align_edge.split('-')[1])
            if src_pos_id[i][e1] != tgt_pos_id[i][e2]:
                pred.append(align_edge)

        for align_edge in gs[i].split():
            e1 = int(align_edge.split('-')[0])
            e2 = int(align_edge.split('-')[1])
            if src_pos_id[i][e1] != tgt_pos_id[i][e2]:
                gold.append(align_edge)

        num_gs += len(gs[i])

        predictions.append(' '.join(list(set(pred))))
        golds.append(' '.join(list(set(gold))))

    for i in range(len(gs)):
        num_same += len(golds[i])

    # f1 = jacana_eval(predictions, golds)
    f1 = simalign_eval(predictions, golds)
    LOG.info('F1 %.3f' % f1)
    print(num_same/num_gs)
    # pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'DET', 'NOUN', 'PRON', 'PROPN', 'PUNCT', 'VERB']
    # f1_mat = np.zeros((17, 17))
    # for x in range(17):
    #     for y in range(17):
    #         # Get pos id
    #         pos1 = pos_tags[x]
    #         pos2 = pos_tags[y]
    #         src_pos_id1, src_pos_id2, tgt_pos_id1, tgt_pos_id2 = [], [], [], []
    #         for i in range(len(source_sentences)):
    #             src_doc = nlp(source_sentences[i])
    #             tgt_doc = nlp(target_sentences[i])
    #             src_pos1, src_pos2, tgt_pos1, tgt_pos2 = [], [], [], []
    #             for token_id in range(len(src_doc)):
    #                 if src_doc[token_id].pos_ == pos1:
    #                     src_pos1.append(token_id)
    #                 if src_doc[token_id].pos_ == pos2:
    #                     src_pos2.append(token_id)
    #             for token_id in range(len(tgt_doc)):
    #                 if tgt_doc[token_id].pos_ == pos1:
    #                     tgt_pos1.append(token_id)
    #                 if tgt_doc[token_id].pos_ == pos2:
    #                     tgt_pos2.append(token_id)
    #             src_pos_id1.append(src_pos1)
    #             src_pos_id2.append(src_pos2)
    #             tgt_pos_id1.append(tgt_pos1)
    #             tgt_pos_id2.append(tgt_pos2)
    #
    #         # POS Evaluation
    #         predictions, golds = [], []
    #         for i in range(len(gs)):
    #             pred, gold = [], []
    #             for align_edge in aligns[i].split():
    #                 e1 = int(align_edge.split('-')[0])
    #                 e2 = int(align_edge.split('-')[1])
    #                 if (e1 in src_pos_id1[i] and e2 in tgt_pos_id2[i]) or (e1 in src_pos_id2[i] and e2 in tgt_pos_id1[i]):
    #                     pred.append(align_edge)
    #             for edge in gs[i].split():
    #                 e1 = int(edge.split('-')[0])
    #                 e2 = int(edge.split('-')[1])
    #                 if (e1 in src_pos_id1[i] and e2 in tgt_pos_id2[i]) or (e1 in src_pos_id2[i] and e2 in tgt_pos_id1[i]):
    #                     gold.append(edge)
    #
    #             predictions.append(' '.join(list(set(pred))))
    #             golds.append(' '.join(list(set(gold))))
    #         f1 = simalign_eval(predictions, golds)
    #         f1_mat[x][y] = f1
    #         LOG.info('%s - %s: %.3f' % (pos1, pos2, f1))
    # return pd.DataFrame(f1_mat)

    # f1_dict = dict()
    # for pos in pos_tags:
    #     src_pos_id, tgt_pos_id = [], []
    #     for i in range(len(source_sentences)):
    #         src_doc = nlp(source_sentences[i])
    #         tgt_doc = nlp(target_sentences[i])
    #         src_pos, tgt_pos = [], []
    #         for token_id in range(len(src_doc)):
    #             if src_doc[token_id].pos_ == pos:
    #                 src_pos.append(token_id)
    #         for token_id in range(len(tgt_doc)):
    #             if tgt_doc[token_id].pos_ == pos:
    #                 tgt_pos.append(token_id)
    #         src_pos_id.append(src_pos)
    #         tgt_pos_id.append(tgt_pos)
    #
    #     # POS Evaluation
    #     predictions, golds = [], []
    #     for i in range(len(gs)):
    #         pred, gold = [], []
    #         for align_edge in aligns[i].split():
    #             e1 = int(align_edge.split('-')[0])
    #             e2 = int(align_edge.split('-')[1])
    #             if e1 in src_pos_id[i] and e2 in tgt_pos_id[i]:
    #                 pred.append(align_edge)
    #         for edge in gs[i].split():
    #             element1 = int(edge.split('-')[0])
    #             element2 = int(edge.split('-')[1])
    #             if element1 in src_pos_id[i] and element2 in tgt_pos_id[i]:
    #                 gold.append(edge)
    #
    #         predictions.append(' '.join(list(set(pred))))
    #         golds.append(' '.join(list(set(gold))))
    #
    #     f1 = simalign_eval(predictions, golds)
    #     LOG.info('POS tag: %s' % pos)
    #     LOG.info('F1 %.3f' % f1)
    #     f1_dict[pos] = [f1]
    # return pd.DataFrame.from_dict(f1_dict)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-dataset', type=str, required=True)
    # parser.add_argument('-model', type=str, required=True)
    # args = parser.parse_args()
    #
    # align_mono_file(args.dataset, args.model)
    # align_mono_file('MTReference', 'simalign-argmax')

    # # POS number analysis
    # df1 = pos_num_analysis('MTReference')
    # df2 = pos_num_analysis('Wikipedia')
    # df3 = pos_num_analysis('Newsela')
    # df4 = pos_num_analysis('arXiv')
    # result = pd.concat([df1, df2, df3, df4], axis=1)
    # result.columns = ['MTRef', 'Wiki', 'Newsela', 'arXiv']
    # print(result)
    # result.to_csv('argmax_pos_analysis.csv')

    result = pos_analysis('MTReference', 'neural-semi-CRF')
    # result.to_csv('itermax.csv')

    # result1, result2 = gs_pos_dist('MTReference')
    # result1.to_csv('gs_pos_analysis.csv')
    # result2.to_csv('gs_pos_num.csv')

