
import sys
from collections import defaultdict
import math
import random
import os
import os.path
import numpy


"""
COMS W4705 - Natural Language Processing - Fall 2020 
Prorgramming Homework 1 - Trigram Language Models
Name: Lu Cheng

"""



def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence




def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  




def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    res = []
    
    sequence.append("STOP")
    
    if n == 1:
        sequence.insert(0, "START")
    else:
        for i in range (0, n-2):
            sequence.insert(0, "START")

        
    for i in range (0, len(sequence) - n + 1):
        res.append(tuple(sequence[i:i+n]))

    return res
   




class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
        self.uniSum = -1
            
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {}

        ##Your code here
        for sentence in corpus:
            unigram = get_ngrams(sentence,1)
            for item in set(unigram): 
                self.unigramcounts[item] = self.unigramcounts.get(item, 0) + unigram.count(item)

            bigram = get_ngrams(sentence,2)
            for item in set(bigram): 
                self.bigramcounts[item] = self.bigramcounts.get(item, 0) + bigram.count(item)

            trigram = get_ngrams(sentence,3)
            for item in trigram: 
                self.trigramcounts[item] = self.trigramcounts.get(item, 0) + trigram.count(item)
        
        
        

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        
        """
        
        if self.uniSum == -1:
            self.uniSum = sum(self.unigramcounts.values()) #- self.unigramcounts[('START',)]
        
        
        if trigram[0] == 'START' and trigram[1] == 'START':
            trigram_prob = self.trigramcounts.get(trigram, 0)/self.unigramcounts[('STOP',)]
            
        elif trigram[:2] in self.bigramcounts.keys():
            trigram_prob = self.trigramcounts.get(trigram, 0)/self.bigramcounts[trigram[:2]]  
            
        else:
            #trigram_prob = self.unigramcounts[tuple([trigram[2]])]/self.uniSum
            trigram_prob = 1/self.uniSum
        
        return trigram_prob
    

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        
        if bigram[0] == 'START':
            bigram_prob = self.bigramcounts.get(bigram, 0)/self.unigramcounts[('STOP',)]
        else:
            bigram_prob = self.bigramcounts.get(bigram, 0)/self.unigramcounts[tuple([bigram[0]])]
               
        return bigram_prob
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        if self.uniSum == -1:
            self.uniSum = sum(self.unigramcounts.values()) #- self.unigramcounts[('START',)]
        

        unigram_prob = self.unigramcounts.get(unigram, 0)/self.uniSum

        return unigram_prob

    # def generate_sentence(self,t=20):
    #     """
    #     COMPLETE THIS METHOD (OPTIONAL)
    #     Generate a random sentence from the trigram model. t specifies the
    #     max length, but the sentence may be shorter if STOP is reached.
    #     """
    #     return result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        smoothed_trigram_prob = lambda1 * self.raw_trigram_probability(trigram) + \
                                lambda2 * self.raw_bigram_probability(trigram[1:3]) + \
                                lambda3 * self.raw_unigram_probability(tuple([trigram[2]]))

        return smoothed_trigram_prob
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        
#         trigram = get_ngrams(sentence, 3)
#         prob = self.smoothed_trigram_probability(trigram)
#         log_prob = {}
#         for words in prob:
#             log_prob[words] = math.log2(prob[words])
        
#         cum_prob = sum(log_prob.values())
        
        cum_prob = 0.0
        trigrams = get_ngrams(sentence, 3)
        for item in trigrams:
            prob = self.smoothed_trigram_probability(item)
            log_prob = math.log2(prob)
            cum_prob = cum_prob + log_prob
        

        return float(cum_prob)

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        
        sum_prob = 0.0
        sen_count = 0.0
        for sentence in corpus:
            sum_prob = sum_prob + self.sentence_logprob(sentence)
            sen_count += (len(sentence)-2)
        
                

        
        perplexity = 2 ** (-sum_prob/sen_count)
        
        

        return float(perplexity) 




def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            total +=1
            pp_model1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            # .. 
            pp_model2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp_model1 < pp_model2:
                correct +=1

        for f in os.listdir(testdir2):
            total +=1
            pp_model2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            # .. 
            pp_model1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            if pp_model2 < pp_model1:
                correct +=1
        
        acc = correct/total
        return acc




if __name__ == "__main__":

    model = TrigramModel('hw1_data/brown_train.txt') 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py ['/Users/chenglu/Desktop/CU/2020\ Fall/COMS\ 4705/hw/hw1_data/ets_toefl_data']

    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt

    
    # Testing perplexity: 
    dev_corpus = corpus_reader('hw1_data/brown_test.txt', model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)
    
    # Essay scoring experiment: 
    acc = essay_scoring_experiment('hw1_data/ets_toefl_data/train_high.txt', 'hw1_data/ets_toefl_data/train_low.txt', 
                                   "hw1_data/ets_toefl_data/test_high", "hw1_data/ets_toefl_data/test_low")
    print(acc)
    
    



