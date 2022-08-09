import regex
import codecs
import collections
import pandas as pd
import itertools
from typing import Dict, List, Tuple, Union
import numpy as np
from numpy import ndarray
from copy import deepcopy
from tqdm import tqdm
from scipy.stats import entropy
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

try:
    import networkx as nx
    from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
except ImportError:
    nx = None
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, XLMModel, XLMTokenizer, RobertaModel, RobertaTokenizer, \
    XLMRobertaModel, XLMRobertaTokenizer, AutoConfig, AutoModel, AutoTokenizer

from utils import get_logger

LOG = get_logger(__name__)


class EmbeddingLoader(object):
    def __init__(self, model: str = "bert-base-multilingual-cased",
                 device=torch.device('mps:0') if torch.backends.mps.is_available() else torch.device('cpu'),
                 layer: int = 8):
        TR_Models = {
            'bert-base-uncased': (BertModel, BertTokenizer),
            'bert-base-multilingual-cased': (BertModel, BertTokenizer),
            'bert-base-multilingual-uncased': (BertModel, BertTokenizer),
            'xlm-mlm-100-1280': (XLMModel, XLMTokenizer),
            'roberta-base': (RobertaModel, RobertaTokenizer),
            'xlm-roberta-base': (XLMRobertaModel, XLMRobertaTokenizer),
            'xlm-roberta-large': (XLMRobertaModel, XLMRobertaTokenizer),
        }

        self.model = model
        self.device = device
        self.layer = layer
        self.emb_model = None
        self.tokenizer = None

        if model in TR_Models:
            model_class, tokenizer_class = TR_Models[model]
            self.emb_model = model_class.from_pretrained(model, output_hidden_states=True)
            self.emb_model.eval()
            self.emb_model.to(self.device)
            self.tokenizer = tokenizer_class.from_pretrained(model)

        else:
            # try to load model with auto-classes
            config = AutoConfig.from_pretrained(model, output_hidden_states=True)
            self.emb_model = AutoModel.from_pretrained(model, config=config)
            self.emb_model.eval()
            self.emb_model.to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        LOG.info("Initialized the EmbeddingLoader with model: {}".format(self.model))

    def get_embed_list(self, sent_batch: List[List[str]]) -> torch.Tensor:
        if self.emb_model is not None:
            with torch.no_grad():
                if not isinstance(sent_batch[0], str):
                    inputs = self.tokenizer(sent_batch, is_split_into_words=True, padding=True, truncation=True,
                                            return_tensors="pt")
                else:
                    inputs = self.tokenizer(sent_batch, is_split_into_words=False, padding=True, truncation=True,
                                            return_tensors="pt")
                hidden = self.emb_model(**inputs.to(self.device))["hidden_states"]
                if self.layer >= len(hidden):
                    raise ValueError(
                        f"Specified to take embeddings from layer {self.layer}, but model has only {len(hidden)} layers.")
                outputs = hidden[self.layer]
                return outputs[:, 1:-1, :]
        else:
            return None


class Simalign:
    def __init__(self, model: str = "bert", token_type: str = "bpe", distortion: float = 0.0,
                 null_align: float = 1.0,
                 matching_methods: str = "mai", device: str = "mps:0" if torch.backends.mps.is_available() else "cpu",
                 layer: int = 8):
        model_names = {
            "bert": "bert-base-multilingual-cased",
            "spanbert": "SpanBERT/spanbert-base-cased"
        }
        all_matching_methods = {"a": "inter", "m": "mwmf", "i": "itermax", "f": "fwd", "r": "rev"}

        self.model = model
        if model in model_names:
            self.model = model_names[model]
        self.token_type = token_type
        self.distortion = distortion
        self.null_align = null_align
        self.matching_methods = all_matching_methods[matching_methods]
        self.device = torch.device(device)

        self.embed_loader = EmbeddingLoader(model=self.model, device=self.device, layer=layer)

        LOG.info(
            "Simalign parameters: model=%s; token_type=%s; matching_methods=%s; device=%s" % (
                model, token_type, matching_methods, device))

    @staticmethod
    def get_max_weight_match(sim: np.ndarray) -> np.ndarray:
        if nx is None:
            raise ValueError("networkx must be installed to use match algorithm.")

        def permute(edge):
            if edge[0] < sim.shape[0]:
                return edge[0], edge[1] - sim.shape[0]
            else:
                return edge[1], edge[0] - sim.shape[0]

        G = from_biadjacency_matrix(csr_matrix(sim))
        matching = nx.max_weight_matching(G, maxcardinality=True)
        matching = [permute(x) for x in matching]
        matching = sorted(matching, key=lambda x: x[0])
        res_matrix = np.zeros_like(sim)
        for edge in matching:
            res_matrix[edge[0], edge[1]] = 1
        return res_matrix

    @staticmethod
    def get_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return (cosine_similarity(X, Y) + 1.0) / 2.0

    @staticmethod
    def get_similarity_norm(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        data = (cosine_similarity(X, Y) + 1.0) / 2.0

        # Normalization
        # _range = np.max(data) - np.min(data)
        # sim_matrix = (data - np.min(data)) / _range

        sim_matrix = data
        sim_matrix[sim_matrix < np.median(sim_matrix)] = 0.
        # sim_matrix[sim_matrix < np.quantile(sim_matrix, 0.75)] = 0.
        return sim_matrix

    @staticmethod
    def average_embeds_over_words(bpe_vectors: List[np.ndarray], word_tokens_pair: List[List[str]]) -> List[np.array]:
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

        new_vectors = []
        for l_id in range(2):
            w_vector = []
            for word_set in w2b_map[l_id]:
                w_vector.append(bpe_vectors[l_id][word_set].mean(0))
            new_vectors.append(np.array(w_vector))
        return new_vectors

    @staticmethod
    def get_alignment_matrix(sim_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, n = sim_matrix.shape
        forward = np.eye(n)[sim_matrix.argmax(axis=1)]  # m x n
        backward = np.eye(m)[sim_matrix.argmax(axis=0)]  # n x m
        return forward, backward.transpose()

    @staticmethod
    def get_alignments_freq(al_matrix, src_spans, tgt_spans):
        m, n = al_matrix.shape
        align_count = collections.defaultdict(lambda: 0)
        for i in range(m):
            for j in range(n):
                if al_matrix[i, j] > 0:
                    # print('{}-{}'.format(src_spans[i], tgt_spans[j]))
                    for src_id in range(len(src_spans[i])):
                        for tgt_id in range(len(tgt_spans[j])):
                            # if len(src_spans[i]) == 1 or len(tgt_spans[j]) == 1:
                            align_count['{}-{}'.format(src_spans[i][src_id], tgt_spans[j][tgt_id])] += (
                                    al_matrix[i, j] / (len(src_spans[i]) * len(tgt_spans[j])))

        return align_count

    @staticmethod
    def apply_distortion(sim_matrix: np.ndarray, ratio: float = 0.5) -> np.ndarray:
        shape = sim_matrix.shape
        if (shape[0] < 2 or shape[1] < 2) or ratio == 0.0:
            return sim_matrix

        pos_x = np.array([[y / float(shape[1] - 1) for y in range(shape[1])] for _ in range(shape[0])])
        pos_y = np.array([[x / float(shape[0] - 1) for x in range(shape[0])] for _ in range(shape[1])])
        distortion_mask = 1.0 - ((pos_x - np.transpose(pos_y)) ** 2) * ratio

        return np.multiply(sim_matrix, distortion_mask)

    @staticmethod
    def get_alignmatrix_greedy(sim_matrix: np.ndarray, src_spans: List[List[int]],
                               tgt_spans: List[List[int]]) -> ndarray:
        m, n = sim_matrix.shape
        alignmatrix = np.zeros((m, n))
        # print(sim_matrix)
        while np.max(sim_matrix) > 0:
            # while np.max(new_sim_matrix) > 0:
            x, y = np.where(sim_matrix == np.max(sim_matrix))
            x, y = int(x[0]), int(y[0])
            alignmatrix[x][y] = 1.
            # print('{}-{}\t{}'.format(src_spans[x], tgt_spans[y], sim_matrix[x][y]))
            sim_matrix[x][y] = 0.

            for e in src_spans[x]:
                for span_id, src_span in enumerate(src_spans):
                    if e in src_span:
                        sim_matrix[span_id] = 0.

            for e in tgt_spans[y]:
                for span_id, tgt_span in enumerate(tgt_spans):
                    if e in tgt_span:
                        sim_matrix[:, span_id] = 0.
        return alignmatrix

    @staticmethod
    def get_combs(src_spans, tgt_spans, df):
        def get_span_dict(spans):
            span_dict = {}
            for span in spans:
                if len(span) == 1:
                    span_dict[span[0]] = list(filter(lambda x: span[0] in x, spans))
            return span_dict

        def deduplicate(mylist):
            new_list = []
            for i in mylist:
                if i not in new_list:
                    new_list.append(i)
            return new_list

        def get_product(pools):
            result = [[]]
            for pool in pools:
                if result == [[]]:
                    result = [x + [y] for x in result for y in pool]
                else:
                    result = [x + [y] for x in result for y in pool if not set(y) & set(x[-1]) or y == x[-1]]
            return [deduplicate(i) for i in result]

        if df.empty:
            return None

        src_span_dict, tgt_span_dict = get_span_dict(src_spans), get_span_dict(tgt_spans)
        if src_span_dict == {} or tgt_span_dict == {}:
            return None

        prod_src, prod_tgt = get_product(src_span_dict.values()), get_product(tgt_span_dict.values())
        combinations = collections.defaultdict(lambda: [])
        for src in prod_src:
            for tgt in prod_tgt:
                if len(src) == len(tgt):
                    sub_df = df.loc[[str(i) for i in src], [str(i) for i in tgt]]
                    sub_nd = sub_df.values
                    # Get the argmax result of the submatrix
                    m, n = sub_nd.shape
                    fwd = np.eye(n)[sub_nd.argmax(axis=1)]
                    bwd = np.eye(m)[sub_nd.argmax(axis=0)]
                    argmax_nd = (fwd * bwd.transpose()) * sub_nd
                    combinations[np.mean(np.max(argmax_nd, axis=1))].extend(
                        [(src[i], tgt[j]) for i, j in zip(np.where(argmax_nd > 0)[0], np.where(argmax_nd > 0)[1])])
                else:
                    continue
        if combinations:
            # for c in sorted(combinations.keys(), reverse=True):
            #     print(combinations[c], c)
            return combinations[sorted(combinations.keys())[-1]]
        else:
            return None

    @staticmethod
    def iter_max(sim_matrix: np.ndarray, max_count: int = 2) -> np.ndarray:
        alpha_ratio = 0.9
        m, n = sim_matrix.shape
        forward = np.eye(n)[sim_matrix.argmax(axis=1)]  # m x n
        backward = np.eye(m)[sim_matrix.argmax(axis=0)]  # n x m
        inter = forward * backward.transpose()

        if min(m, n) <= 2:
            return inter

        new_inter = np.zeros((m, n))
        count = 1
        while count < max_count:
            mask_x = 1.0 - np.tile(inter.sum(1)[:, np.newaxis], (1, n)).clip(0.0, 1.0)
            mask_y = 1.0 - np.tile(inter.sum(0)[np.newaxis, :], (m, 1)).clip(0.0, 1.0)
            mask = ((alpha_ratio * mask_x) + (alpha_ratio * mask_y)).clip(0.0, 1.0)
            mask_zeros = 1.0 - ((1.0 - mask_x) * (1.0 - mask_y))
            if mask_x.sum() < 1.0 or mask_y.sum() < 1.0:
                mask *= 0.0
                mask_zeros *= 0.0

            new_sim = sim_matrix * mask
            fwd = np.eye(n)[new_sim.argmax(axis=1)] * mask_zeros
            bac = np.eye(m)[new_sim.argmax(axis=0)].transpose() * mask_zeros
            new_inter = fwd * bac

            if np.array_equal(inter + new_inter, inter):
                break
            inter = inter + new_inter
            count += 1
        return inter

    @staticmethod
    def gather_null_aligns(sim_matrix: np.ndarray, inter_matrix: np.ndarray) -> List[float]:
        shape = sim_matrix.shape
        if min(shape[0], shape[1]) <= 2:
            return []
        norm_x = normalize(sim_matrix, axis=1, norm='l1')
        norm_y = normalize(sim_matrix, axis=0, norm='l1')

        entropy_x = np.array([entropy(norm_x[i, :]) / np.log(shape[1]) for i in range(shape[0])])
        entropy_y = np.array([entropy(norm_y[:, j]) / np.log(shape[0]) for j in range(shape[1])])

        mask_x = np.tile(entropy_x[:, np.newaxis], (1, shape[1]))
        mask_y = np.tile(entropy_y, (shape[0], 1))

        all_ents = np.multiply(inter_matrix, np.minimum(mask_x, mask_y))
        return [x.item() for x in np.nditer(all_ents) if x.item() > 0]

    @staticmethod
    def apply_percentile_null_aligns(sim_matrix: np.ndarray, ratio: float = 1.0) -> np.ndarray:
        shape = sim_matrix.shape
        if min(shape[0], shape[1]) <= 2:
            return np.ones(shape)
        norm_x = normalize(sim_matrix, axis=1, norm='l1')
        norm_y = normalize(sim_matrix, axis=0, norm='l1')
        entropy_x = np.array([entropy(norm_x[i, :]) / np.log(shape[1]) for i in range(shape[0])])
        entropy_y = np.array([entropy(norm_y[:, j]) / np.log(shape[0]) for j in range(shape[1])])
        mask_x = np.tile(entropy_x[:, np.newaxis], (1, shape[1]))
        mask_y = np.tile(entropy_y, (shape[0], 1))

        ents_mask = np.where(np.minimum(mask_x, mask_y) > ratio, 0.0, 1.0)

        return ents_mask

    @staticmethod
    def get_span_index(src_sents: List[str], tgt_sents: List[str],
                       max_d: int = 3) -> Tuple[List[List[list]], List[List[list]]]:
        src_spans, tgt_spans = [], []
        for sent_id in range(len(src_sents)):
            src_sent_idx = list(range(len(src_sents[sent_id].split())))
            tgt_sent_idx = list(range(len(tgt_sents[sent_id].split())))
            src_span_idx, tgt_span_idx = [], []
            for d in range(1, max_d + 1):
                src_span_idx.extend(
                    [src_sent_idx[i: i + d] for i in range(0, len(src_sent_idx)) if i + d <= len(src_sent_idx)])
                tgt_span_idx.extend(
                    [tgt_sent_idx[i: i + d] for i in range(0, len(tgt_sent_idx)) if i + d <= len(tgt_sent_idx)])
            src_spans.append(src_span_idx)
            tgt_spans.append(tgt_span_idx)
        return src_spans, tgt_spans

    @staticmethod
    def get_bpe_index(bpe_map: List[List[List[int]]], src_idx: List[List[list]], tgt_idx: List[List[list]],
                      reverse=False) -> List[List[List[list]]]:
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

    @staticmethod
    def average_embeds_over_spans(bpe_vectors: List[np.ndarray],
                                  span_tokens_pair: Union[List[List[list]], List[List[List[list]]]]) -> List[np.array]:
        w2b_map = span_tokens_pair

        new_vectors = []
        for l_id in range(2):
            span_vector = []
            for span_set in w2b_map[l_id]:
                span_vector.append(bpe_vectors[l_id][span_set].mean(0))
            new_vectors.append(np.array(span_vector))
        return new_vectors

    def align_spans_iter(self, source_sentences, target_sentences, batch_size=100):
        device = torch.device(self.device)

        words_tokens = []
        for sent_id in range(len(source_sentences)):
            l1_tokens = [self.embed_loader.tokenizer.tokenize(word) for word in source_sentences[sent_id].split()]
            l2_tokens = [self.embed_loader.tokenizer.tokenize(word) for word in target_sentences[sent_id].split()]
            words_tokens.append([l1_tokens, l2_tokens])

        sentences_bpe_lists = []
        sentences_b2w_map = []
        for sent_id in range(len(words_tokens)):
            sent_pair = [[bpe for w in sent for bpe in w] for sent in words_tokens[sent_id]]
            b2w_map_pair = [[i for i, w in enumerate(sent) for _ in w] for sent in words_tokens[sent_id]]
            sentences_bpe_lists.append(sent_pair)
            sentences_b2w_map.append(b2w_map_pair)

        # Get all possible spans (len <= 3)
        source_spans, target_spans = self.get_span_index(source_sentences, target_sentences)
        spans_pair_bpe = self.get_bpe_index(sentences_b2w_map, source_spans, target_spans)

        ds = [(idx, source_sentences[idx], target_sentences[idx]) for idx in range(len(source_sentences))]
        data_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
        aligns = []
        for batch_id, batch_sentences in enumerate(tqdm(data_loader)):
            batch_sentences[1], batch_sentences[2] = list(batch_sentences[1]), list(batch_sentences[2])
            batch_vectors_src = self.embed_loader.get_embed_list(batch_sentences[1])
            batch_vectors_tgt = self.embed_loader.get_embed_list(batch_sentences[2])
            # Normalization
            batch_vectors_src = F.normalize(batch_vectors_src, dim=2)
            batch_vectors_tgt = F.normalize(batch_vectors_tgt, dim=2)

            batch_vectors_src = batch_vectors_src.cpu().detach().numpy()
            batch_vectors_tgt = batch_vectors_tgt.cpu().detach().numpy()

            for in_batch_id, sent_id in enumerate(batch_sentences[0].numpy()):
                sent_pair = sentences_bpe_lists[sent_id]
                vectors = [batch_vectors_src[in_batch_id, :len(sent_pair[0])],
                           batch_vectors_tgt[in_batch_id, :len(sent_pair[1])]]
                src_len, tgt_len = len(source_sentences[sent_id].split()), len(target_sentences[sent_id].split())
                src_spans, tgt_spans = source_spans[sent_id], target_spans[sent_id]
                vectors = self.average_embeds_over_spans(vectors, spans_pair_bpe[sent_id])
                sim = self.get_similarity_norm(vectors[0], vectors[1])  # 1/x n*m -> n*m
                sim_df = pd.DataFrame(sim, index=[str(i) for i in src_spans],
                                      columns=[str(i) for i in tgt_spans])
                # print(sim_df)

                # Baseline1: Greedy method
                alignment_matrix = self.get_alignmatrix_greedy(sim, src_spans, tgt_spans)
                span_scores = collections.defaultdict(lambda: [])
                for i in range(len(vectors[0])):
                    for j in range(len(vectors[1])):
                        if alignment_matrix[i, j] > 0.:
                            for x in source_spans[sent_id][i]:
                                for y in target_spans[sent_id][j]:
                                    span_scores['{}-{}'.format(x, y)].append(sim[i, j])

                # # Baseline2: Cover the words as much as possible
                # forward, reverse = self.get_alignment_matrix(sim)
                # alignment_matrix = forward * reverse
                # span_scores = collections.defaultdict(lambda: [])
                # index_s, column_s, src_id, tgt_id = set(), set(), set(), set()
                # for i in range(src_len):
                #     for j in range(tgt_len):
                #         if alignment_matrix[i, j] > 0.:
                #             src_spans = list(filter(lambda x: i not in x, src_spans))
                #             tgt_spans = list(filter(lambda x: j not in x, tgt_spans))
                #
                #             for src_span in source_spans[sent_id]:
                #                 if i in src_span:
                #                     index_s.add(str(src_span))
                #                     src_id.add(source_spans[sent_id].index(src_span))
                #             for tgt_span in target_spans[sent_id]:
                #                 if j in tgt_span:
                #                     column_s.add(str(tgt_span))
                #                     tgt_id.add(target_spans[sent_id].index(tgt_span))
                #             span_scores['{}-{}'.format(i, j)].append(sim[i, j])
                # # print(span_scores)
                # # delete useless rows/columns of dataframe
                # sim_df = sim_df.drop(index=list(index_s), columns=list(column_s))
                # sim_df = sim_df.loc[:, (sim_df != 0).any(axis=0)]
                # sim_df = sim_df[sim_df.apply(np.sum, axis=1) != 0]
                # # delete useless rows/columns of ndarray
                # sim = np.delete(sim, tuple(src_id), axis=0)
                # sim = np.delete(sim, tuple(tgt_id), axis=1)
                # src_spans = [src_spans[i] for i in range(len(src_spans)) if
                #              i not in [int(j) for j in np.argwhere(np.all(sim[..., :] == 0, axis=1))]]
                # tgt_spans = [tgt_spans[i] for i in range(len(tgt_spans)) if
                #              i not in [int(j) for j in np.argwhere(np.all(sim[..., :] == 0, axis=0))]]
                # # sim = np.delete(sim, np.argwhere(np.all(sim[..., :] == 0, axis=1)), axis=0)
                # # sim = np.delete(sim, np.argwhere(np.all(sim[..., :] == 0, axis=0)), axis=1)
                # # print(sim_df)
                # combination = self.get_combs(src_spans, tgt_spans, sim_df)
                # # print(combination)
                # if combination is not None:
                #     for comb in combination:
                #         for src in comb[0]:
                #             for tgt in comb[1]:
                #                 # if str([src]) not in sim_df.index or str([tgt]) not in sim_df.columns or sim_df.at[str([src]), str([tgt])] > 0:
                #                 span_scores['{}-{}'.format(src, tgt)].append(None)

                aligns.append(' '.join(sorted([F"{p}" for p, vals in span_scores.items()],
                                              key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))))
        return aligns

    def align_spans_freq(self, source_sentences, target_sentences, batch_size=100):
        device = torch.device(self.device)

        words_tokens = []
        for sent_id in range(len(source_sentences)):
            l1_tokens = [self.embed_loader.tokenizer.tokenize(word) for word in source_sentences[sent_id].split()]
            l2_tokens = [self.embed_loader.tokenizer.tokenize(word) for word in target_sentences[sent_id].split()]
            words_tokens.append([l1_tokens, l2_tokens])

        sentences_bpe_lists = []
        sentences_b2w_map = []
        for sent_id in range(len(words_tokens)):
            sent_pair = [[bpe for w in sent for bpe in w] for sent in words_tokens[sent_id]]
            b2w_map_pair = [[i for i, w in enumerate(sent) for _ in w] for sent in words_tokens[sent_id]]
            sentences_bpe_lists.append(sent_pair)
            sentences_b2w_map.append(b2w_map_pair)

        # Get all possible spans (len <= 3)
        source_spans, target_spans = self.get_span_index(source_sentences, target_sentences)
        spans_pair_bpe = self.get_bpe_index(sentences_b2w_map, source_spans, target_spans)

        ds = [(idx, source_sentences[idx], target_sentences[idx]) for idx in range(len(source_sentences))]
        data_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
        aligns = []
        for batch_id, batch_sentences in enumerate(tqdm(data_loader)):
            batch_sentences[1], batch_sentences[2] = list(batch_sentences[1]), list(batch_sentences[2])
            batch_vectors_src = self.embed_loader.get_embed_list(batch_sentences[1])
            batch_vectors_tgt = self.embed_loader.get_embed_list(batch_sentences[2])
            # Normalization
            batch_vectors_src = F.normalize(batch_vectors_src, dim=2)
            batch_vectors_tgt = F.normalize(batch_vectors_tgt, dim=2)

            batch_vectors_src = batch_vectors_src.cpu().detach().numpy()
            batch_vectors_tgt = batch_vectors_tgt.cpu().detach().numpy()

            for in_batch_id, sent_id in enumerate(batch_sentences[0].numpy()):
                sent_pair = sentences_bpe_lists[sent_id]
                vectors = [batch_vectors_src[in_batch_id, :len(sent_pair[0])],
                           batch_vectors_tgt[in_batch_id, :len(sent_pair[1])]]
                vectors = self.average_embeds_over_spans(vectors, spans_pair_bpe[sent_id])
                sim = self.get_similarity(vectors[0], vectors[1])
                src_len, tgt_len = len(source_sentences[sent_id].split()), len(target_sentences[sent_id].split())
                # for i in range(len(vectors[0])):
                #     for j in range(len(vectors[1])):
                #         if not ((i >= src_len and j < tgt_len) or (i < src_len and j >= tgt_len)):
                #             sim[i][j] *= 0.9
                # print(sim)
                # for i in range(len(vectors[0])):
                #     for j in range(len(vectors[1])):
                #         if not (i < src_len and j < tgt_len):
                #         # if (len(source_spans[sent_id][i]) == 2 and len(target_spans[sent_id][j]) == 3) or (
                #         #         len(source_spans[sent_id][i]) == 3 and len(target_spans[sent_id][j]) == 2):
                #             sim[i][j] *= 0.9
                forward, reverse = self.get_alignment_matrix(sim)
                # alignment_matrix = forward * reverse
                # alignment_matrix = self.iter_max(src_len, tgt_len, sim)
                alignment_matrix = forward * 0.5 + reverse * 0.5

                # print(alignment_matrix)

                span_scores = self.get_alignments_freq(alignment_matrix, source_spans[sent_id], target_spans[sent_id])

                # span_scores = collections.defaultdict(lambda: [])
                # for i in range(len(vectors[0])):
                #     for j in range(len(vectors[1])):
                #         # if forward[i, j] > 0:
                #         #     print('forward {} - {}'.format(source_spans[sent_id][i], target_spans[sent_id][j]))
                #         # if reverse[i, j] > 0:
                #         #     print('reverse {} - {}'.format(source_spans[sent_id][i], target_spans[sent_id][j]))
                #         if alignment_matrix[i, j] > 0:
                #             # print('{} - {}'.format(source_spans[sent_id][i], target_spans[sent_id][j]))
                #             if len(source_spans[sent_id][i]) == 1 or len(target_spans[sent_id][j]) == 1:
                #                 for x in source_spans[sent_id][i]:
                #                     for y in target_spans[sent_id][j]:
                #                         span_scores['{}-{}'.format(x, y)].append(sim[i, j])
                aligns.append(' '.join(sorted([F"{p}" for p, vals in span_scores.items() if vals >= 1.0],
                                              key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))))
                # print(span_scores)
                # aligns.append(' '.join(sorted([F"{p}" for p, vals in span_scores.items()])))

        return aligns

    def align_spans_bidirection(self, source_sentences, target_sentences, batch_size=100):
        device = torch.device(self.device)

        words_tokens = []
        for sent_id in range(len(source_sentences)):
            l1_tokens = [self.embed_loader.tokenizer.tokenize(word) for word in source_sentences[sent_id].split()]
            l2_tokens = [self.embed_loader.tokenizer.tokenize(word) for word in target_sentences[sent_id].split()]
            words_tokens.append([l1_tokens, l2_tokens])

        sentences_bpe_lists = []
        sentences_b2w_map = []
        for sent_id in range(len(words_tokens)):
            sent_pair = [[bpe for w in sent for bpe in w] for sent in words_tokens[sent_id]]
            b2w_map_pair = [[i for i, w in enumerate(sent) for _ in w] for sent in words_tokens[sent_id]]
            sentences_bpe_lists.append(sent_pair)
            sentences_b2w_map.append(b2w_map_pair)

        # Get all possible spans
        source_spans, target_spans = self.get_span_index(source_sentences, target_sentences)
        source_words, target_words = [], []
        for src_sent, tgt_sent in zip(source_spans, target_spans):
            src_words, tgt_words = [], []
            src_words.extend([span for span in src_sent if len(span) == 1])
            tgt_words.extend([span for span in tgt_sent if len(span) == 1])
            source_words.append(src_words)
            target_words.append(tgt_words)

        s2t_spans_pair_bpe = self.get_bpe_index(sentences_b2w_map, source_spans, target_words)
        t2s_spans_pair_bpe = self.get_bpe_index(sentences_b2w_map, target_spans, source_words, True)

        # alignments = []
        s2t_aligns, t2s_aligns = [], []
        for spans_pair_bpe in [s2t_spans_pair_bpe, t2s_spans_pair_bpe]:
            ds = [(idx, source_sentences[idx], target_sentences[idx]) for idx in range(len(source_sentences))]
            data_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
            # aligns = []
            for batch_id, batch_sentences in enumerate(tqdm(data_loader)):
                batch_sentences[1], batch_sentences[2] = list(batch_sentences[1]), list(batch_sentences[2])
                batch_vectors_src = self.embed_loader.get_embed_list(batch_sentences[1])
                batch_vectors_tgt = self.embed_loader.get_embed_list(batch_sentences[2])
                # Normalization
                batch_vectors_src = F.normalize(batch_vectors_src, dim=2)
                batch_vectors_tgt = F.normalize(batch_vectors_tgt, dim=2)

                batch_vectors_src = batch_vectors_src.cpu().detach().numpy()
                batch_vectors_tgt = batch_vectors_tgt.cpu().detach().numpy()

                for in_batch_id, sent_id in enumerate(batch_sentences[0].numpy()):
                    sent_pair = sentences_bpe_lists[sent_id]
                    if spans_pair_bpe == t2s_spans_pair_bpe:
                        vectors = [batch_vectors_tgt[in_batch_id, :len(sent_pair[1])],
                                   batch_vectors_src[in_batch_id, :len(sent_pair[0])]]
                        row, column = target_spans, source_words
                    else:
                        vectors = [batch_vectors_src[in_batch_id, :len(sent_pair[0])],
                                   batch_vectors_tgt[in_batch_id, :len(sent_pair[1])]]
                        row, column = source_spans, target_words
                    vectors = self.average_embeds_over_spans(vectors, spans_pair_bpe[sent_id])
                    sim = self.get_similarity(vectors[0], vectors[1])

                    # src_len, tgt_len = len(source_sentences[sent_id].split()), len(target_sentences[sent_id].split())
                    # for i in range(len(vectors[0])):
                    #     for j in range(len(vectors[1])):
                    #         if not (i < src_len and j < tgt_len):
                    #         # if (len(source_spans[sent_id][i]) == 2 and len(target_spans[sent_id][j]) == 3) or (
                    #         #         len(source_spans[sent_id][i]) == 3 and len(target_spans[sent_id][j]) == 2):
                    #             sim[i][j] *= 0.9

                    forward, reverse = self.get_alignment_matrix(sim)
                    alignment_matrix = reverse
                    # alignment_matrix = self.get_alignmatrix_iter(sim, row[sent_id], column[sent_id])
                    # alignment_matrix = self.iter_max(sim)
                    # print(alignment_matrix)
                    # alignment_matrix = forward * 0.5 + reverse * 0.5
                    # print(alignment_matrix)
                    span_scores = collections.defaultdict(lambda: [])
                    # span_scores = collections.defaultdict(lambda: 0)
                    for i in range(len(vectors[0])):
                        for j in range(len(vectors[1])):
                            if alignment_matrix[i, j] > 0:

                                # if spans_pair_bpe == t2s_spans_pair_bpe:
                                #     print('{} - {}'.format(column[sent_id][j], row[sent_id][i]))
                                # else:
                                #     print('{} - {}'.format(row[sent_id][i], column[sent_id][j]))

                                for x in row[sent_id][i]:
                                    for y in column[sent_id][j]:
                                        if spans_pair_bpe == t2s_spans_pair_bpe:
                                            span_scores['{}-{}'.format(y, x)].append(sim[i, j])
                                            # span_scores['{}-{}'.format(y, x)] += (alignment_matrix[i, j] / (len(row[sent_id][i]) * len(column[sent_id][j])))
                                        else:
                                            span_scores['{}-{}'.format(x, y)].append(sim[i, j])
                                            # span_scores['{}-{}'.format(x, y)] += (alignment_matrix[i, j] / (len(row[sent_id][i]) * len(column[sent_id][j])))
                    # aligns.append(' '.join(sorted([F"{p}" for p, vals in span_scores.items() if vals >= 1.0])))
                    # print(span_scores)
                    if spans_pair_bpe == t2s_spans_pair_bpe:
                        # alignments[sent_id] += (' ' + ' '.join(sorted([F"{p}" for p, vals in span_scores.items()])))
                        t2s_aligns.append(' '.join(sorted([F"{p}" for p, vals in span_scores.items()])))
                    else:
                        # alignments.append(' '.join(sorted([F"{p}" for p, vals in span_scores.items() if vals])))
                        s2t_aligns.append(' '.join(sorted([F"{p}" for p, vals in span_scores.items()])))
                    # print(alignments)
        aligns = [' '.join(sorted([i for i in s2t_aligns[sent_id].split() if i in t2s_aligns[sent_id].split()],
                                  key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))) for sent_id in
                  range(len(s2t_aligns))]
        # aligns = [' '.join(sorted(list(set(alignments[i].split())), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))) for i in range(len(alignments))]
        return aligns

    def align_sentences(self, source_sentences, target_sentences, batch_size=100):
        convert_to_words = (self.token_type == "word")
        device = torch.device(self.device)

        words_tokens = []
        for sent_id in range(len(source_sentences)):
            l1_tokens = [self.embed_loader.tokenizer.tokenize(word) for word in source_sentences[sent_id].split()]
            l2_tokens = [self.embed_loader.tokenizer.tokenize(word) for word in target_sentences[sent_id].split()]
            words_tokens.append([l1_tokens, l2_tokens])

        sentences_bpe_lists = []
        sentences_b2w_map = []
        for sent_id in range(len(words_tokens)):
            sent_pair = [[bpe for w in sent for bpe in w] for sent in words_tokens[sent_id]]
            b2w_map_pair = [[i for i, w in enumerate(sent) for _ in w] for sent in words_tokens[sent_id]]
            sentences_bpe_lists.append(sent_pair)
            sentences_b2w_map.append(b2w_map_pair)

        corpora_lengths = [len(source_sentences), len(target_sentences)]

        ds = [(idx, source_sentences[idx], target_sentences[idx]) for idx in range(len(source_sentences))]
        data_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
        aligns = []
        for batch_id, batch_sentences in enumerate(tqdm(data_loader)):
            batch_sentences[1], batch_sentences[2] = list(batch_sentences[1]), list(batch_sentences[2])
            batch_vectors_src = self.embed_loader.get_embed_list(batch_sentences[1])
            batch_vectors_trg = self.embed_loader.get_embed_list(batch_sentences[2])
            btach_sim = None
            if not convert_to_words:
                batch_vectors_src = F.normalize(batch_vectors_src, dim=2)
                batch_vectors_trg = F.normalize(batch_vectors_trg, dim=2)

                btach_sim = torch.bmm(batch_vectors_src, torch.transpose(batch_vectors_trg, 1, 2))
                btach_sim = ((btach_sim + 1.0) / 2.0).cpu().detach().numpy()

            batch_vectors_src = batch_vectors_src.cpu().detach().numpy()
            batch_vectors_trg = batch_vectors_trg.cpu().detach().numpy()

            for in_batch_id, sent_id in enumerate(batch_sentences[0].numpy()):
                sent_pair = sentences_bpe_lists[sent_id]
                vectors = [batch_vectors_src[in_batch_id, :len(sent_pair[0])],
                           batch_vectors_trg[in_batch_id, :len(sent_pair[1])]]

                if not convert_to_words:
                    sim = btach_sim[in_batch_id, :len(sent_pair[0]), :len(sent_pair[1])]
                else:
                    vectors = self.average_embeds_over_words(vectors, words_tokens[sent_id])
                    sim = self.get_similarity(vectors[0], vectors[1])

                all_mats = {}

                sim = self.apply_distortion(sim, self.distortion)

                all_mats["fwd"], all_mats["rev"] = self.get_alignment_matrix(sim)
                all_mats["inter"] = all_mats["fwd"] * all_mats["rev"]
                if "mwmf" in self.matching_methods:
                    all_mats["mwmf"] = self.get_max_weight_match(sim)
                if "itermax" in self.matching_methods:
                    all_mats["itermax"] = self.iter_max(sim)

                raw_aligns = []
                b2w_aligns = set()
                raw_scores = collections.defaultdict(lambda: [])
                b2w_scores = collections.defaultdict(lambda: [])
                log_aligns = []

                for i in range(len(vectors[0])):
                    for j in range(len(vectors[1])):
                        ext = self.matching_methods
                        if all_mats[ext][i, j] > 0:
                            raw_aligns.append('{}-{}'.format(i, j))
                            raw_scores['{}-{}'.format(i, j)].append(sim[i, j])
                            if self.token_type == "bpe":
                                b2w_aligns.add(
                                    '{}-{}'.format(sentences_b2w_map[sent_id][0][i], sentences_b2w_map[sent_id][1][j]))
                                b2w_scores['{}-{}'.format(sentences_b2w_map[sent_id][0][i],
                                                          sentences_b2w_map[sent_id][1][j])].append(sim[i, j])
                                if ext == "inter":
                                    log_aligns.append('{}-{}:({}, {})'.format(i, j, sent_pair[0][i], sent_pair[1][j]))
                            else:
                                b2w_aligns.add('{}-{}'.format(i, j))

                if convert_to_words:
                    aligns.append(' '.join(sorted([F"{p}" for p, vals in raw_scores.items()])))
                    # aligns.append(' '.join(sorted([F"{p}-{str(round(np.mean(vals), 3))[1:]}" for p, vals in raw_scores.items()])))
                else:
                    aligns.append(' '.join(sorted([F"{p}" for p, vals in b2w_scores.items()],
                                                  key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))))
                    # aligns.append(' '.join(sorted([F"{p}-{str(round(np.mean(vals), 3))[1:]}" for p, vals in b2w_scores.items()])))
        return aligns
