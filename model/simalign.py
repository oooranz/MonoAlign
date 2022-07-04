import regex
import codecs
import collections
from typing import Dict, List, Tuple, Union
import numpy as np
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
    def __init__(self, model: str = "bert-base-multilingual-cased", device=torch.device('cpu'), layer: int = 8):
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
                                            return_tensors="pt", return_token_type_ids=True)
                else:
                    inputs = self.tokenizer(sent_batch, is_split_into_words=False, padding=True, truncation=True,
                                            return_tensors="pt", return_token_type_ids=True)
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
                 matching_methods: str = "mai", device: str = "cpu", layer: int = 8):
        model_names = {
            "bert": "bert-base-multilingual-cased",
            "roberta": "roberta-base"
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
            "Simalign parameters: model=%s; token_type=%s; distortion=%s; null_align=%s; matching_methods=%s; device=%s" % (
                model, token_type, distortion, null_align, matching_methods, device))

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

    # TODO: adjust to the model 'roberta'
    @staticmethod
    def average_embeds_over_words(bpe_vectors: np.ndarray, word_tokens_pair: List[List[str]]) -> List[np.array]:
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
    def apply_distortion(sim_matrix: np.ndarray, ratio: float = 0.5) -> np.ndarray:
        shape = sim_matrix.shape
        if (shape[0] < 2 or shape[1] < 2) or ratio == 0.0:
            return sim_matrix

        pos_x = np.array([[y / float(shape[1] - 1) for y in range(shape[1])] for _ in range(shape[0])])
        pos_y = np.array([[x / float(shape[0] - 1) for x in range(shape[0])] for _ in range(shape[1])])
        distortion_mask = 1.0 - ((pos_x - np.transpose(pos_y)) ** 2) * ratio

        return np.multiply(sim_matrix, distortion_mask)

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
                    aligns.append(' '.join(sorted([F"{p}" for p, vals in b2w_scores.items()])))
                    # aligns.append(' '.join(sorted([F"{p}-{str(round(np.mean(vals), 3))[1:]}" for p, vals in b2w_scores.items()])))
        return aligns
