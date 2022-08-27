import os
import codecs
import argparse
import model
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, required=True)
    parser.add_argument('-model', type=str, required=True)
    args = parser.parse_args()

    align_mono_file(args.dataset, args.model)
    # align_mono_file('MTReference', 'simalign-argmax')
