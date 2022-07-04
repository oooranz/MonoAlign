#!/usr/bin/env python3
import codecs
import argparse
import os.path
import os

class Fastalign:
    def __init__(self, symmetrization='gdfa', add_prob=False):
        self.symmetrization = symmetrization
        self.add_prob = add_prob

    def align_sentences(self, source_sentences, target_sentences, out_path='fastalign'):
        fastalign_path = "/mounts/Users/student/masoud/tools/fast_align/build/fast_align"
        fastalign_prob_path = "/mounts/Users/student/masoud/tools/fast_align/build_prob/fast_align"
        atools_path = "/mounts/Users/student/masoud/tools/fast_align/build/atools"

        # create parallel text
        paral_path = out_path + ".txt"

        fa_file = codecs.open(paral_path, "w", "utf-8")
        for source_sentence, target_sentence in zip(source_sentences, target_sentences):
            fa_file.write(source_sentence + " ||| " + target_sentence + "\n")
        fa_file.close()

        # FastAlign
        if not self.add_prob:
            os.system("{} -i {} -v -d -o > {}.fwd".format(fastalign_path, paral_path, out_path))
            os.system("{} -i {} -v -d -o -r > {}.rev".format(fastalign_path, paral_path, out_path))
        # FastAlign + Prob
        else:
            os.system("{} -i {} -v -d -o > {}.fwd".format(fastalign_prob_path, paral_path, out_path))
            os.system("{} -i {} -v -d -o -r > {}.rev".format(fastalign_prob_path, paral_path, out_path))

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
            if not self.add_prob:
                for l1, l2 in zip(f1.readlines(), f2.readlines()):
                    l1 = set(l1.strip().split())
                    l2 = set(l2.strip().split())
                    aligns.append(' '.join(sorted([x for x in l1 & l2])))
            else:
                for l1, l2 in zip(f1.readlines(), f2.readlines()):
                    l1 = {x[:x.rfind("-")]: round(float(x[x.rfind("-")+1:]), 3) for x in l1.strip().split()}
                    l2 = {x[:x.rfind("-")]: round(float(x[x.rfind("-")+1:]), 3) for x in l2.strip().split()}
                    inters = [F"{x}-{round((l1[x]+l2[x])/2, 3)}" for x in set(l1) & set(l2)]
                    aligns.append(' '.join(sorted(inters)))
        elif self.symmetrization == 'union':
            f1 = open(out_path + ".fwd", "r")
            f2 = open(out_path + ".rev", "r")
            if not self.add_prob:
                for l1, l2 in zip(f1.readlines(), f2.readlines()):
                    l1 = set(l1.strip().split())
                    l2 = set(l2.strip().split())
                    aligns.append(' '.join(sorted([x for x in l1 | l2])))
            else:
                for l1, l2 in zip(f1.readlines(), f2.readlines()):
                    l1 = {x[:x.rfind("-")]: round(float(x[x.rfind("-")+1:]), 3) for x in l1.strip().split()}
                    l2 = {x[:x.rfind("-")]: round(float(x[x.rfind("-")+1:]), 3) for x in l2.strip().split()}
                    inters = []
                    for x in set(l1) | set(l2):
                        if x in l1 and x in l2:
                            inters.append("{}-{:f}".format(x, (l1[x]+l2[x])/2))
                        elif x in l1 and x not in l2:
                            inters.append("{}-{:f}".format(x, l1[x]/2))
                        elif x not in l1 and x in l2:
                            inters.append("{}-{:f}".format(x, l2[x]/2))
                    aligns.append(' '.join(sorted(inters)))
        
        os.system("rm {}.txt".format(out_path))
        os.system("rm {}.fwd".format(out_path))
        os.system("rm {}.rev".format(out_path))
        os.system("rm {}.gdfa".format(out_path))
        
        return aligns

