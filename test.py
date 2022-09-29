import re
import spacy
import pandas as pd
from spacy.tokenizer import Tokenizer
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import palettable


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# special_cases = {":)": [{"ORTH": ":)"}]}
# prefix_re = re.compile(r'''''')
# suffix_re = re.compile(r'''''')
# infix_re = re.compile(r'''''')
# simple_url_re = re.compile(r'''''')
#
# def custom_tokenizer(nlp):
#     return Tokenizer(nlp.vocab, rules=special_cases,
#                      prefix_search=prefix_re.search,
#                      suffix_search=suffix_re.search,
#                      infix_finditer=infix_re.finditer,
#                      url_match=simple_url_re.match)
#
# nlp = spacy.load("en_core_web_sm")
# nlp.tokenizer = custom_tokenizer(nlp)
# src_doc = nlp('after the signing ceremony , president akaev and premier li peng answered the questions from reporters respectively .')
# tgt_doc = nlp('after the signing ceremonym president akaev and premier li peng answered questions of reporters .')
# src_pos, tgt_pos = [], []
#
# for token_id in range(len(src_doc)):
#     src_pos.append(src_doc[token_id].pos_)
# for token_id in range(len(tgt_doc)):
#     tgt_pos.append(tgt_doc[token_id].pos_)
#
# print(src_pos)
# print(tgt_pos)
df1 = pd.read_csv('c_pos_analysis.csv', header=None)
df1 = df1.iloc[1:, 1:]
df2 = pd.read_csv('i_pos_analysis.csv', header=None)
df2 = df2.iloc[1:, 1:]
a = df1.values - df2.values
# print(np.mean(a)/2)
df = pd.DataFrame(a)

# df = (df - df.min()) / (df.max() - df.min())
# df = (df-df.mean())/(df.std())
# df = pd.DataFrame(normalization(df.values))
# print(np.sum(np.diagonal(df.values))/np.sum(df.values))

df3 = pd.read_csv('gs_pos_num.csv', header=None)
df3 = df3.iloc[1:, 1:]
df3 = np.tril(df3)
# print(np.sum(np.diagonal(df3))/np.sum(df3))

mask = np.triu(np.ones_like(df, dtype=bool), k=1)
# mask[df == 1.] = True

df4 = pd.read_csv('gs_pos_num.csv', header=None)
df4 = df4.iloc[1:, 1:]
print(np.sum(df4.values, axis=0))

# print(mask)
# mat = df.values
pos_tags = ['ADP', 'DET', 'NOUN', 'PUNCT', 'PROPN', 'CCONJ', 'VERB', 'ADV', 'SCONJ', 'PART', 'ADJ', 'PRON', 'AUX',
            'NUM', 'SYM', 'INTJ', 'X']
plt.figure(dpi=200)
# ax = sns.heatmap(df, cmap='YlGnBu', xticklabels=pos_tags, yticklabels=pos_tags)
# cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
ax = sns.heatmap(df, mask=mask, square=True, xticklabels=pos_tags, yticklabels=pos_tags, cmap=cmap, vmin=-1, vmax=1)
# ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

# plt.title('title', loc='left', fontsize=18)

# plt.show()
