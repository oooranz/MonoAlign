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

aligner = model.Simalign(matching_methods='a')
source_sentences = ['summer in us this year proved to be one of the hottest summers in history .']
target_sentences = ['the summer in us this year is one of the hottest in history .']

aligns = aligner.align_sentences(source_sentences, target_sentences)
print(aligns)
