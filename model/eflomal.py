#!/usr/bin/env python3
import codecs
import argparse
import os.path
import os

class Eflomal:
    def __init__(self, model='3', symmetrization='gdfa'):
        self.model = model
        self.symmetrization = symmetrization

    def align_sentences(self, source_sentences, target_sentences, out_path='eflomal'):
        eflomal_path = "/mounts/Users/student/linpq/Documents/NLP/Alignment/eflomal/"
        atools_path = "/mounts/Users/student/masoud/tools/fast_align/build/atools"

        # create parallel text
        paral_path = out_path + ".txt"

        fa_file = codecs.open(paral_path, "w", "utf-8")
        for source_sentence, target_sentence in zip(source_sentences, target_sentences):
            fa_file.write(source_sentence + " ||| " + target_sentence + "\n")
        fa_file.close()

        os.system(eflomal_path + "align.py -i {0} --model {1} -f {2}.fwd -r {2}.rev --overwrite".format(paral_path, self.model, out_path))

        aligns = []
        if self.symmetrization == 'gdfa':
            os.system("{0} -i {1}.fwd -j {1}.rev -c grow-diag-final-and > {1}.gdfa".format(atools_path, out_path))
            f = open(out_path + ".gdfa", "r")
            for l in f.readlines():
                l = set(l.strip().split())
                aligns.append(' '.join(sorted([x for x in l])))
        elif self.symmetrization == 'inter':
            f1 = open(out_path + ".fwd", "r")
            f2 = open(out_path + ".rev", "r")
            for l1, l2 in zip(f1.readlines(), f2.readlines()):
                l1 = set(l1.strip().split())
                l2 = set(l2.strip().split())
                aligns.append(' '.join(sorted([x for x in l1 & l2])))
        elif self.symmetrization == 'union':
            f1 = open(out_path + ".fwd", "r")
            f2 = open(out_path + ".rev", "r")
            for l1, l2 in zip(f1.readlines(), f2.readlines()):
                l1 = set(l1.strip().split())
                l2 = set(l2.strip().split())
                aligns.append(' '.join(sorted([x for x in l1 | l2])))

        os.system("rm {}.txt".format(out_path))
        os.system("rm {}.fwd".format(out_path))
        os.system("rm {}.rev".format(out_path))
        os.system("rm {}.gdfa".format(out_path))
        
        return aligns

