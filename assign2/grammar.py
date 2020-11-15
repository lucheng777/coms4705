"""
COMS W4705 - Natural Language Processing - Fall 20 
Homework 2 - Parsing with Context Free Grammars 
Name: Lu Cheng
Uni: lc3452
"""

import sys
from collections import defaultdict
from math import fsum
import string

class Pcfg(object):
    """
    Represent a probabilistic context free grammar.
    """

    def __init__(self, grammar_file):
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None
        self.read_rules(grammar_file)

    def read_rules(self,grammar_file):

        for line in grammar_file:
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line:
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else:
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()


    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1)
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False.
        """
        # TODO, Part 1
        lhs_to_rules = self.lhs_to_rules
        
        for item in lhs_to_rules:
            rule = lhs_to_rules[item]
            sum = []
            a = 0
            b = 0
            for e in rule:
                
                if len(e[1]) == 1 : 
                    test = e[1][0]
                    if test.islower() or test in string.punctuation or test.isdigit():
                        pass
                        
                elif len(e[1]) == 2:
                    test1 =e[1][0]
                    test2 = e[1][1]
                    if test1.isupper() and test2.isupper():
                        pass
                else:
                    sys.stderr.write('the rules does not have the right format')
                
                sum.append(e[2])
            if round(fsum(sum),2) != 1.0:
                sys.stderr.write('all probabilities for the same lhs symbol does not sum to 1.0')
                return False
        
        
        return True


 
if __name__ == "__main__":
    #with open(sys.argv[1],'r') as grammar_file:
    with open('atis3.pcfg','r') as grammar_file:
        grammar = Pcfg(grammar_file)
        print(grammar.verify_grammar())

