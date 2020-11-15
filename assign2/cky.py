"""
COMS W4705 - Natural Language Processing - Fall 2020  
Homework 2 - Parsing with Context Free Grammars 
Name: Lu Cheng
Uni: lc3452
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        grammar = self.grammar
        
        # number of tokens
        n = len(tokens)
        # generate keys of the dictionary
        index_list = list(itertools.product(range(0, n+1), repeat=2))
        index = [index_list[i:i+n+1] for i in range(0, len(index_list), n+1)]
        # default table represented as distionary
        table = defaultdict(dict)
        
        ## cky algo
        # initialization
        for i in range(n):
            ## another way for generating keys of the dictionay
            # ind = (i, i + 1) 
            ind = index[i][i+1]
            rules = grammar.rhs_to_rules[(tokens[i],)]
            for rule in rules:
                table[ind][rule[0]] = tokens[i]
                
        # main loop
        for length in range(2, n+1):
            for i in range(n-length+1):
                j = i + length
                ind_ij = index[i][j]
                for k in range(i+1, j):
                    ind_ik = index[i][k]
                    ind_kj = index[k][j]
                    if len(table[ind_ik]) is not 0 and len(table[ind_kj]) is not 0:
                        unions = list(itertools.product(table[ind_ik], table[ind_kj]))
                        for union in unions:
                            if union in grammar.rhs_to_rules:
                                nonterminal = grammar.rhs_to_rules[union]
                                for nt in nonterminal:
                                    table[ind_ij][nt[0]] = union
                                    
        if grammar.startsymbol in table[index[0][n]]:
            return True
            
        return False 
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        grammar = self.grammar
        
        table= defaultdict(dict)
        probs = defaultdict(dict)
        n = len(tokens) 
        

        
        # initialization
        for i in range(n):
            key = (i, i + 1)
            if (tokens[i],) in grammar.rhs_to_rules:
                rules = grammar.rhs_to_rules[(tokens[i],)]
                for rule in rules:
                    table[key][rule[0]] = tokens[i]
                    probs[key][rule[0]] = math.log2(rule[2])
                
        # main loop
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                key = (i, j)
                for k in range(i + 1, j):
                    if len(table[(i, k)]) is not 0 and len(table[(k, j)]) is not 0:
                        unions = list(itertools.product(list(table[(i, k)].keys()), list(table[(k, j)].keys())))
                        for union in unions:
                            if union in grammar.rhs_to_rules:
                                nonterminal = grammar.rhs_to_rules[union]
                                for nt in nonterminal:
                                    if nt[0] not in table[key]:
                                        table[key][nt[0]] = ((union[0], i, k), (union[1], k, j))
                                        probs[key][nt[0]] = math.log2(nt[2]) + probs[(i, k)][union[0]] + probs[(k, j)][union[1]]
                                    else:
                                        if probs[key][nt[0]] < math.log2(nt[2]) + probs[(i, k)][union[0]] + probs[(k, j)][union[1]]:
                                            table[key][nt[0]] = ((union[0], i, k), (union[1], k, j))
                                            probs[key][nt[0]] = math.log2(nt[2]) + probs[(i, k)][union[0]] + probs[(k, j)][union[1]]

        
        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4

    if j - i == 1:
        res = (nt, chart[(i, j)][nt])
        return res
    
    res1 = get_tree(chart, chart[(i, j)][nt][0][1], chart[(i, j)][nt][0][2], chart[(i, j)][nt][0][0])
    res2 = get_tree(chart, chart[(i, j)][nt][1][1], chart[(i, j)][nt][1][2], chart[(i, j)][nt][1][0])
    return (nt, res1, res2)
 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.'] 
        print(parser.is_in_language(toks))
        table,probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)
        
