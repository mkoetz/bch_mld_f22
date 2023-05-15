# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:02:34 2022

@author: mkoetz1
"""

import numpy as np
import numpy.linalg as nla
import random
import galois
from itertools import product
from itertools import combinations

# construct matrix as GF([[row1],[row2],...,[rowk]])

# example
# f2 = galois.GF(2)
# H = f2([[1,0,1,0],[0,1,1,0]])

# need to prefix every vector/matrix/etc. with galois field

# list of tuples: vs = [f2(x) for x in product([0,1], repeat=3)]
# eh? H = f2(np.array([v for v in vs]))
# f2(f2([1,1,0,1,0,0,1])@H) ### @ is how to multiply matrices in np

# for i in range(11):
#     noise = np.random.randint(2,size=7)
#     syndrome = f2(f2(noise)@H)
#     print(syndrome)

# better random generator:
# from numpy.random import default_rng
# rng = default_rng()
# rng_ints = rng.integers
# noise = f2(rng_ints(2,size=7))
# can also handle normal/other distributions
rng = np.random.default_rng()
rng_ints = rng.integers

# generate the codewords
# f27 = [F2(x) for x in product([0,1],repeat=7)]
# HC = []
# for v in f27:
#     if np.array_equal(v@np.transpose(H),np.zeros(3,dtype=int)): ## <--- better
#     if all(F2(F2(v)@np.transpose(H))==[0,0,0]):
#         HC.append(F2(v))
        
F2 = galois.GF(2)


def weight(v):
    return np.count_nonzero(v)

# need to expand this to nonbinary Hamming codes eventually
def HammCode(p):
    # check matrix for length 2^p-1 Hamming code
    # length = 2^p - 1, dimension = n-p?
#    F2 = galois.GF(2)
    vs = [F2(x) for x in product([0,1],repeat=p)]
    vs.remove(vs[0])
    H = F2(np.array([v for v in vs]))
    return np.transpose(H)

# nonbinary Hamming code attempts
# prefer(?) poly representation of GF(p^r): F = galois.GF(p**r,display='poly')
# vs = [F(x) for x in product(list(F.elements),repeat=r)] <-- r = self.power?
# need to remove scalar multiples (linear dependence among columns) first
# find first nonzero element in vector, only take those with a 1
#
# important alias: a = F.primitive_element
#
# with a list of vectors vs, we can do this?
# leading_ones = []
# for v in vs:
#     if v[np.argmax(v!=0)] == 1:
#         leading_ones.append(v)
# H = F(np.array([v for v in leading_ones]))

# NEED TO ADD MINIMUM DISTANCE FOR HAMMING CODES
# wait, no, it's always 3
class BinHamm:
    def __init__(self,power):
        self.length = 2**power - 1
#        self.field=galois.GF(field)
        self.power = power
        self.dim = self.length-self.power

        vs = [galois.GF(2)(x) for x in product([0,1],repeat=power)]
        vs.remove(vs[0])
        self.pcheck = np.transpose(galois.GF(2)(np.array([v for v in vs])))
        self.genmat = self.pcheck.left_null_space()

        syndromes = {}
        syndromes[str(galois.GF(2)(np.zeros(self.length-self.dim,dtype=int)))] = galois.GF(2)(np.zeros(self.length,dtype=int))
        # take advantage of the covering radius being 1 to shortcut syndrome table
        for e in range(self.length):
            error_vec = galois.GF(2)(np.zeros(self.length,dtype=int))
            error_vec[e] = galois.GF(2)(1)
            s = str(galois.GF(2)(error_vec@np.transpose(self.pcheck)))
            if s not in syndromes:
                syndromes[s] = error_vec
        self.syndromeTable = syndromes
        
        # should be able to define encode and decode, maybe generator matrix, list of codewords functions

# BinHamm(z)=pHamm(2,1,z)
class pHamm: #takes prime, power, and redundancy
    def __init__(self,prime,power,red):
        self.length = int((prime**(power*red) - 1)/(prime**power - 1))
        self.dim = self.length - red
        self.field = galois.GF(prime**power, display='poly')
# way too slow
#        self.vector_space = [self.field(x) for x in product(list(self.field.elements),repeat=self.length)]
        
        vs = [self.field(x) for x in product(list(self.field.elements),repeat=red)]
        leading_ones = []
        for v in vs:
            if v[np.argmax(v!=0)] ==1:
                leading_ones.append(v)
        self.pcheck = np.transpose(self.field(np.array([v for v in leading_ones])))
        self.genmat = np.transpose(self.pcheck).left_null_space()

        syndromes = {}
        syndromes[str(self.field(np.zeros(self.length-self.dim,dtype=int)))] = self.field(np.zeros(self.length,dtype=int))
        # take advantage of the covering radius being 1 to shortcut syndrome table
        # THIS WILL NOT WORK FOR NONBINARY CODES...or maybe it will?
        for e in range(self.length):
            error_vec = self.field(np.zeros(self.length,dtype=int))
            error_vec[e] = self.field(1)
            s = str(self.field(error_vec@np.transpose(self.pcheck)))
            if s not in syndromes:
                syndromes[s] = error_vec
        self.syndromeTable = syndromes

# should I generate codewords upon creation? seems...slow

# can I create a class based on a parity check matrix? generator matrix?
# convert between the two?

# .row_space() to get...codewords? nope - returns matrix

# random (0,1)-matrix: np.mod(np.random.permutation(4*8).reshape(4,8),2)

# for now, pass ground field to code classes
class codeFromCheck:
    def __init__(self,check_mat,ground_field):
        self.pcheck = check_mat
        self.field = ground_field
        self.genmat = np.transpose(self.pcheck).left_null_space()
        self.length = len(check_mat[0])
        self.dim = self.length - len(check_mat) #assumes full rank, but not guaranteed, look into np.linalg.matrix_rank
        # need to figure out an appropriate weight here
        self.syndromeTable = syn_table(self.length,self.dim,3,self.pcheck)
        
class codeFromGenerator:
    def __init(self,gen_mat,ground_field):
        self.gen = gen_mat
        self.field = ground_field
        self.pcheck = np.transpose(self.gen).left_null_space()
        self.length = len(gen_mat[0])
        self.dim = len(gen_mat)
        # need to figure out an appropriate weight here
        self.syndromeTable = syn_table(self.length,self.dim,2,self.pcheck)
        
#ok, so...error loop, compare messages, record results for plotting...

# this loop tries random vectors until it finds a codeword
# super not efficient
# for i in range(20):
#     try_count=0
#     while True:
#         v = random.choice(fullspace)
#         try_count += 1
#         if all(v@np.transpose(C.pcheck)==F2(np.zeros(4,dtype=int))):
#             break
#     print(try_count)

# currently only works on pHamm
def random_codeword(code):
#    vs = code.vector_space
    # codewords = []
    # for v in vs:
    #     if all(v@np.transpose(code.pcheck) == F2(np.zeros(code.length-code.dim,dtype=int))):
    #         codewords.append(v)
    # while True:
    #     v = random.choice(vs)
    #     if all(v@np.transpose(code.pcheck) == F2(np.zeros(code.length-code.dim,dtype=int))):
    #         return  v
# now making use of generator matrix from left_null_space() method
    vs = [code.field(x) for x in product(list(code.field.elements),repeat=code.dim)]
    return random.choice(vs)@code.genmat

# setting up binary Golay code
Golay2_H_rows = [[1,0,0,1,1,1,0,0,0,1,1,1],
                 [1,0,1,0,1,1,0,1,1,0,0,1],
                 [1,0,1,1,0,1,1,0,1,0,1,0],
                 [1,0,1,1,1,0,1,1,0,1,0,0],
                 [1,1,0,0,1,1,1,0,1,1,0,0],
                 [1,1,0,1,0,1,1,1,0,0,0,1],
                 [1,1,0,1,1,0,0,1,1,0,1,0],
                 [1,1,1,0,0,1,0,1,0,1,1,0],
                 [1,1,1,0,1,0,1,0,0,0,1,1],
                 [1,1,1,1,0,0,0,0,1,1,0,1],
                 [0,1,1,1,1,1,1,1,1,1,1,1]]
Golay2_H = F2(np.concatenate((np.array([v for v in Golay2_H_rows]).T,
                              np.identity(11,dtype=int))).T)

# setting up ternary Golay code
F3 = galois.GF(3)
Golay3_H_rows = [[1,1,1,2,2,0,1,0,0,0,0],
                 [1,1,2,1,0,2,0,1,0,0,0],
                 [1,2,1,0,1,2,0,0,1,0,0],
                 [1,2,0,1,2,1,0,0,0,1,0],
                 [1,0,2,2,1,1,0,0,0,0,1]]
Golay3_H = F3(np.array([v for v in Golay3_H_rows]))

# packaging 'galois' BCH implementation
def binary_bch(n,k):
    valid_dims = [p[1] for p in galois.bch_valid_codes(n)]
    if k not in valid_dims:
        raise ValueError("No valid code with those parameters.")
    return galois.BCH(n,k)

# note that n-k must be even
def reed_solomon(n,k):
    return galois.ReedSolomon(n, k)

# n=length, w=weight
def qary_noise(n,w,q):
    field = galois.GF(q,display='poly')
    noise = np.zeros(n,dtype=int)
#    noise[:w] = field.Random() ### CAN GIVE 0
    i = 0
    while w > 0:
        r = field.Random()
        if r != 0:
            noise[i] = r
            w -= 1
            i += 1
    np.random.shuffle(noise)
    return field(noise)

# for i in range(10):
#     q = rng_ints(16)
#     word = HC[q]
#     noise = qary_noise(7,1,2)
#     sent = word + noise
#     syn = sent@np.transpose(H)
#     print(syn)

# =============================================================================
# For now, manual syndrome table for Hamming codes (and others)
# maybe accept that I have to know the minimum distance ahead of time, run
# through all possible errors of weight <= (d-1)/2, compute syndromes.
# What about larger errors? Return "not decodable" or guess? For larger
# weight errors, I can compute the syndrome of one and call it a day, I guess.
# =============================================================================

# generate a syndrome table for errors of weight 1
# syn_table = {}
# syn_table[str(F2(np.zeros(3,dtype=int)))] = F2(np.zeros(7,dtype=int))
# for i in range(7):
#     syn = F2(np.zeros(7,dtype=int))
#     syn[i]=F2(1)
#     syndrome = F2(syn@np.transpose(H))
#     syn_table[str(syndrome)]=syn


# computes minimum weight(?) error vectors for each possible syndrome
# if error has larger weight, will not decode correctly, as expected
# for syndrome decoding
def syn_table(length,dim,max_wt,check_mat):
    syndromes = {}
    syndromes[str(F2(np.zeros(length-dim,dtype=int)))] = F2(np.zeros(length,dtype=int))
    for w in range(max_wt+1):
        error_support = [list(x) for x in combinations(range(length),w)]
        for e in error_support:
            error_vec = F2(np.zeros(length,dtype=int))
            error_vec[e] = F2(1)
            s = F2(error_vec@np.transpose(check_mat))
            if str(s) not in syndromes:
                syndromes[str(s)] = error_vec
    return syndromes
    
# generate noise, pick random codeword, add, compute syndrome, decode, check
# currently this only works for binary Hamming codes, see below for BCH, RS codes
# for i in range(10):
#     noise = qary_noise(7,1,2)
#     c = random.choice(HC)
#     transmit = F2(noise+c)
#     s = F2(transmit@np.transpose(H))
#     decode = F2(transmit + syn_table[str(s)])
#     print(c==decode)

# for i in range(10):
#     noise = qary_noise(7,rng_ints(3),2)
#     c = random.choice(HC)
#     transmit = F2(noise+c)
#     decode = F2(transmit + syn_table[str(F2(transmit@np.transpose(H)))])
#     if any(c!=decode):
#         print(c,noise,decode)

# C = binary_bch(15,7)
# for i in range(10):
#     m = F2.Random(C.k) # random message to encode, not a codeword
#     c = C.encode(m)
#     noise = qary_noise(15,rng_ints(2),2)
#     transmit = F2(c+noise)
#     decode = C.decode(transmit) # decodes to m, not to c
#     if any (m!=decode):
#         print(c,noise,decode)

# C = reed_solomon(n,k)
# for i in range(10):
    # field = C.field
#     m = field.Random(C.k) # random message to encode
#     c = C.encode(m)
#     noise = qary_noise(15,rng_ints(3),q)
#     transmit = field(c+noise)
#     decode = C.decode(transmit)
#     if any(m!=decode):
#         print(m,noise,decode)



# even in Sage, a lookup table is generated by looping through all
# "acceptible" errors, i.e., up to a certain max_wt
# decoding is fast after this (according to that manual)

import matplotlib.pyplot as plt


# HC = pHamm(2,1,3)
# # generate the codewords
# f27 = [F2(x) for x in product([0,1],repeat=7)]
# HC_words = []
# for v in f27:
#     if np.array_equal(v@np.transpose(HC.pcheck),np.zeros(3,dtype=int)): ## <--- better
#     # if all(F2(F2(v)@np.transpose(H))==[0,0,0]):
#         HC_words.append(F2(v))

# code must have ground field available
# working on binary first
def codewords(code):
#    full_space = [code.field(x) for x in product(code.field.elements, repeat = code.length)]
    valid_words = [code.field(x)@code.genmat for x in product(code.field.elements, repeat = code.dim)]
### AWFUL, SLOW - use genmat instead
    # valid_words = []
    # for v in full_space:
    #     if np.array_equal(v@np.transpose(code.pcheck),np.zeros(code.length-code.dim,dtype=int)):
    #         valid_words.append(v)
    return valid_words

def min_wt(code):
    mw = code.length
    for w in codewords(code):
        if weight(w) < mw and weight(w)>0:
            mw = weight(w)
    return mw

#pass codewords as parameter to save computations

def error_run(code,w,tries):
    possible_messages = codewords(code)
    successes = 0
    failures = 0
    for i in range(tries):
        c = random.choice(possible_messages)
        noise = qary_noise(code.length,w,2)
        transmit = c + noise
        s = transmit@np.transpose(code.pcheck)
        decode = transmit + code.syndromeTable[str(s)]
        if all(c == decode):
            successes += 1
        else:
            failures += 1
    return successes, failures

def error_run_no_syn(code,w,st,tries):
    possible_messages = codewords(code)
    successes = 0
    failures = 0
    # d = min_wt(code)
    # table_wt = int(np.ceil((d-1)/2))+1
    # st = syn_table(code.length, code.dim, w, code.pcheck)
    for i in range(tries):
        c = random.choice(possible_messages)
        noise = qary_noise(code.length,w,2)
        transmit = c + noise
        s = transmit@np.transpose(code.pcheck)
        decode = transmit + st[str(s)]
        if all(c == decode):
            successes += 1
        else:
            failures += 1
    return successes, failures

def code_trial(code, max_err_wt, tries, how_many):
    #possible_messages = codewords(code)
    st = syn_table(code.length, code.dim, max_err_wt, code.pcheck)
    error_weights = range(1,max_err_wt+1)
    correctly_decoded = np.zeros(max_err_wt)
    upper_errors = np.zeros(max_err_wt)
    lower_errors = np.zeros(max_err_wt)
    for w in error_weights:
        percents_for_run = []
        for i in range(how_many):
            (s,f) = error_run_no_syn(code,w,st,tries)
            percents_for_run.append(s/(s+f))
        correctly_decoded[w-1] = np.average(percents_for_run)
        upper_errors[w-1] = np.max(percents_for_run)-correctly_decoded[w-1]
        lower_errors[w-1] = correctly_decoded[w-1] - np.min(percents_for_run)
    error_bars = [lower_errors,upper_errors]
    return error_weights, correctly_decoded, error_bars

# T = code_trial(C, 4, 20, 20)
# plt.errorbar(T[0],T[1],yerr=T[2])

# tracking decode error weights
def error_run2(code,w,tries):
    possible_messages = codewords(code)
    successes = 0
    failures = 0
    error_weights = []
    for i in range(tries):
        c = random.choice(possible_messages)
        noise = qary_noise(code.length,w,2)
        transmit = c + noise
        s = transmit@np.transpose(code.pcheck)
        decode = transmit + code.syndromeTable[str(s)]
        if all(c == decode):
            successes += 1
            error_weights.append([w,0])
        else:
            failures += 1
            error_weights.append([w,weight(decode-c)])
    return successes, failures#, error_weights

def error_run_bch(code,w,tries):
#    possible_messages = codewords(code)
    successes = 0
    failures = 0
    error_weights = []
    for i in range(tries):
        m = galois.GF(2).Random(code.k)
        c = code.encode(m)
        noise = qary_noise(code.n,w,2)
        transmit = c + noise
        d = code.decode(transmit)
        if all(m == d):
            successes += 1
            error_weights.append([w,0])
        else:
            failures += 1
            error_weights.append([w,weight(d-m)]) # except this should relate to codewords, not messages
    return successes, failures, error_weights

def visualize_error_rate(k,n,trials,runs):
    H = galois.GF(2)(np.mod(np.random.permutation(k*n).reshape(k,n),2))
    C = codeFromCheck(H, galois.GF(2))
    d = min_wt(C)
    w = int(np.floor((d-1)/2))
    T = code_trial(C, w+2, trials, runs)
    fig = plt.figure()
    plt.errorbar(T[0],T[1],T[2])
    return T,fig
# bit errors = weight(decoded - transmitted)
# BER = errors/length, which I can control so far

# x = [1,2,3]
# y = [2,-1,4]
# plt.title('first plot')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.plot(x,y)
# plt.show()

# for i in range(50):
#     trial = error_run(C,3,50)
#     pc = trial[0]/(trial[0]+trial[1])
#     weight3_trials.append(pc)
    

# np.average(weight3_trials)
# Out[147]: 0.26839999999999997

# rates_to_plot.append((3,0.2684))

# x = [2,3]

# y = [0.8116,0.2684]

# yerr = [np.max(percent_correct)-np.min(percent_correct),np.max(weight3_trials)-np.min(weight3_trials)]

# plt.errorbar(x,y,yerr=yerr)