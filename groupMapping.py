# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 21:35:22 2021

@author: kesson
"""
import numpy as np
from bitarray import bitarray
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator
from copy import copy 
import time
def label2Pauli(s): # can be imported from groupedFermionicOperator.py
    """
    Convert a Pauli string into Pauli object. 
    Note that the qubits are labelled in descending order: 'IXYZ' represents I_3 X_2 Y_1 Z_0
    
    Args: 
        s (str) : string representation of a Pauli term
    
    Returns:
        qiskit.quantum_info.Pauli: Pauli object of s
    """
    
    xs = []
    zs = []
    label2XZ = {'I': (0, 0), 'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
    for c in s[::-1]:
        x, z = label2XZ[c]
        xs.append(x)
        zs.append(z)
    return Pauli(z = zs, x = xs)

def str2WeightedPaulis(s):
	s = s.strip()
	IXYZ = ['I', 'X', 'Y', 'Z']
	prev_idx = 0
	coefs = []
	paulis = []
	is_coef = True
	for idx, c in enumerate(s + '+'):
		if idx == 0: continue
		if is_coef and c in IXYZ:
			coef = complex(s[prev_idx : idx].replace('i', 'j'))
			coefs.append(coef)
			is_coef = False
			prev_idx = idx
		if not is_coef and c in ['+', '-']:
			label = s[prev_idx : idx]
			paulis.append(label2Pauli(label))
			is_coef = True
			prev_idx = idx
	return WeightedPauliOperator([[c, p] for (c,p) in zip(coefs, paulis)])


def complete_mapping(naive_mapping):
    mapping = copy(naive_mapping)
    for k, w in naive_mapping.items():
        k1, k2 = k
        if (k1 != k2) and ((k2, k1) not in naive_mapping.keys()):
            new_k = (k2, k1)
            paulis = list(map(lambda x: [np.conj(x[0]), x[1]], w.paulis))
            mapping[new_k] = WeightedPauliOperator(paulis)
    return mapping

def bitmasks(n,m):
    """
    (a math tool)
    Return a list of numbers with `m` ones in their binary expressions in ascending order. 
    The last element in the list does not exceed `until`.
    
    Args:
        n (int) : number of all numbers
        m (int) : number of ones
    
    Returns:
        ([Int]) : the generated list
    """
    if m < n:
        if m > 0:
            for x in bitmasks(n-1,m-1):
                yield bitarray([1]) + x
            for x in bitmasks(n-1,m):
                yield bitarray([0]) + x
        else:
            yield n * bitarray('0')
    else:
        yield n * bitarray('1')
        
def to10(C,n):
    C10=[]
    for i in C:
        tn=n
        tv=0
        for j in i:
            tn-=1
            if j=='1':
                tv+=2**tn
        C10.append(tv)
    return np.array(C10)

def to2(MAPC,n):
    C2=[]
    for i in MAPC:
        C="{0:b}".format(i)
        C=C.zfill(n)
        C2.append(C)
    return C2

def gop(n):
    """
    (a math tool)
    Return a list of numbers with `n` ones in their binary expressions in ascending order. 
    The last element in the list does not exceed `until`.
    
    Args:
        n (int) : number of ones
        until (int) : upper bound of list elements
    
    Returns:
        ([Int]) : the generated list
    """
    OP=[]
    for i in range(n):
        for j in range(i+1,n):
            OP.append(2**i+2**j)
    return np.array(OP)

def coset(CV,GOP):
    ALL=[]
    for i in GOP:
        SUB=[]
        Ctmp=CV^i
        for j in range(len(CV)):
            if CV[j] in Ctmp:
                k=np.where(Ctmp==CV[j])[0][0]
                if k>j:
                    SUB.append([j,k])
        ALL.append(SUB)
    return ALL

def Bextend(Basis,bn):
    B=np.array(Basis)
    for i in range(bn-1):
        C=B[i]^B[(i+1):]
        Basis.extend(list(C))        

def cosetop(bn,sn):
    Basis=[]
    for i in range(bn):
        Basis.append(2**i)
        
    B=np.array(Basis)
    Bextend(Basis,len(Basis))
    if len(B)==sn-1:
        return np.array(Basis),bn
    
    breaker=1
    while(breaker):
        for i in range(2**bn):
            sum=0
            for j in B^i:
                sum+=(j in Basis)
            if sum==0:
                B=np.insert(B,len(B),i)
                break
            if i==2**bn-1:
                breaker=0
        Basis=list(B)
        # print(Basis)
        Bextend(Basis,len(Basis))
      
        # subop=np.array(Basis)
        # sum=0
        # for i in subop^1:
        #     if i in subop:
        #         sum+=1
      
        if len(B)==sn-1:
            return np.array(Basis),bn
    return cosetop(bn+1,sn)

def mapping(subgroup,subop,num):
    map01=np.zeros(num,dtype=int)
    k=0
    m=1
    while sum(map01==0)!=1:
        for i in range(k+1,num):
            tmp=[k,i]
            for j in range(len(subgroup)):
                if (tmp in subgroup[j]) and (map01[i]==0):
                    map01[i]=int(map01[k])^subop[j]
                    break
        k+=1
        if(k>num):
            map01=np.zeros(num,dtype=int)
            map01[0]=m
            m+=1
            k=0
    return map01

def signmap(C2, subgroup, CV,C):
    sign=[]
    for l in subgroup:
        signtmp=[0]*(2**len(C2[0]))
        for i in l:
            tmp=''
            for j in range(len(C2[i[0]])):
                k1=C2[i[0]][j]
                k2=C2[i[1]][j]
                pt=CV[i[0]]^CV[i[1]]
                A="{0:b}".format(pt)
                for s in range(1,len(A)):
                    if A[-s]=='1':
                        break
                    else:
                        pt=pt^2**(s-1)
                        
                parity=(-1)**((bin(CV[i[0]]&((2**(len(bin(pt))-2)-1)^pt)).count('1'))%2)
                if(pt==0):
                    parity=1
                if (k1=='0' and k2=='0') or (k1=='1' and k2=='0'):
                    tmp+='0'
                elif (k1=='1' and k2=='1') or (k1=='0' and k2=='1'):
                    tmp+='1'
                    
            signtmp+=parity*maker(tmp)/2**len(C2[0])
        sign.append(signtmp)
    return sign
"""
A:I+Z X+iY
B:I-Z X-iY
(X+Y)(X-Y)(I+Z)(I-Z)
->
ABAB
D=A(x)B(x)A(x)B
"""
def maker(k):
    A=[1,1]
    B=[1,-1]
    D=[1]
    for i in k:
        if i=='0':
            D=np.kron(D,A)
        elif i=='1':
            D=np.kron(D,B)    
    return D

"""
(X+Y)(X-Y)(I+Z)(I-Z)
->
QQNN
D=Q(0,1)(x)Q(0,1)(x)N(0,1)(x)N(0,1)
opmaker(sign_matrix,where is Q+, where is Q-, number of qubit)
"""
def opmaker(sign,subop,n):
    result=[]
    for i in range(len(subop)):
        E=[]
        signtmp=np.where(sign[i])
        # signtmp=np.arange(2**n)
        binop="{0:b}".format(subop[i])
        binop=binop.zfill(n)
        for k in signtmp[0]:
            bins="{0:b}".format(k)
            bins=bins.zfill(n)
            tmp=''
            icount=0
            for j in range(len(binop)):
                j1=binop[j]
                j2=bins[j]
                if j1=='0' and j2=='0':
                    tmp+='I'
                elif j1=='0' and j2=='1':
                    tmp+='Z'
                elif j1=='1' and j2=='0':
                    tmp+='X'
                else:
                    icount+=1
                    tmp+='Y'
        # i*i=-1 only i print i
            if icount%2==0 and icount!=0:
                if (-1)**(icount//2)<0:
                    tmp='-'+tmp
            elif icount%2==1:
                tmp='i'+tmp
                if (-1)**(icount//2)<0:
                    tmp='-'+tmp
            E.append(tmp)   
        result.append(np.array(E))
    return result

def getmap(n,m):
    C=[]
    fer={}
    for b in bitmasks(n,m):
        C.append(b.to01())

    C.reverse()
    numbit=int(np.log2(len(C)))
    print('Initializing:')
    t1=time.time()
    CV=to10(C,n)
    GOP=gop(n)
    subgroup=coset(CV,GOP)
    t2=time.time()
    print('time cost',t2-t1)
    print('guess qubits:',numbit)
    t1=time.time()
    subop,numbit=cosetop(numbit,n)
    t2=time.time()
    print('group preserving qubits:',numbit,'time cost',t2-t1)
    MAPC=mapping(subgroup,subop,len(C))
    C2=to2(MAPC,numbit)
    sign=signmap(C2, subgroup, CV,C)
    op=opmaker(sign,subop,numbit)
    for i in range(len(sign)):
        sign[i]=sign[i][sign[i]!=0]

    num=0
    for i in range(n):
        for j in range(i+1,n):
            key= (j,i)
            value=''
            for k in range(len(sign[num])):
                if op[num][k][0]=='-':
                    sign[num][k]=-sign[num][k]
                    op[num][k]=op[num][k][1:]
                if sign[num][k]>0:
                    value+='+'+str(sign[num][k])+op[num][k]
                else:
                    value+=str(sign[num][k])+op[num][k]
                    
            fer[key]=str2WeightedPaulis(value)
            num+=1

    subgroup0=[]
    for i in range(n):
        tmp=np.where(CV==CV|2**i)[0]
        stmp=[]
        for j in tmp:
            stmp.append([j,j])
        subgroup0.append(stmp)
        
    subop0=np.array([0]*n)
    sign0=signmap(C2, subgroup0, CV,C)
    op0=opmaker(sign0,subop0,numbit)
    for i in range(len(sign0)):
        sign0[i]=sign0[i][sign0[i]!=0]
    
    for i in range(n):
        key= (i,i)
        value=''
        for k in range(len(sign0[i])):
            if sign0[i][k]>0:
                value+='+'+str(sign0[i][k])+op0[i][k]
            else:
                value+=str(sign0[i][k])+op0[i][k]
                    
        fer[key]=str2WeightedPaulis(value)
    fer=complete_mapping(fer)
#    print("fer end")
    return fer#,C,C2,subop,subgroup,op,sign

# state=8
# electron=2
# ####       
# fer=getmap(state,electron)
