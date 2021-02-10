from qiskit.chemistry import FermionicOperator
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli

import numpy as np
from fermionic2QubitMapping import fermionic2QubitMapping

def kDelta(i, j):
    return 1 * (i == j)

def label2Pauli(s):
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


class groupedFermionicOperator:
    """
    An alternative representation (grouped-operator form) of `qiskit.chemistry.FermionicOperator`.
    Two-electron terms (a_p^ a_q^ a_r a_s) are rearranged into products of (a_i^ a_j).
    (a_n^, a_n are creation and annihilation operators, respectively, acting on the n-th qubit.)
    """
    def __init__(self, ferOp, num_electron, labeling_method=None):
        """
        This class rewrites a `FermionicOperator` into a grouped-operator form stored in `self.grouped_op`.
        The `self.grouped_op` is a dictionary containing information of all one- and two-electron terms.
        For a one-electron term h_pq=val, it is stored as {(p, q): val}.
        For a two-electron term h_pqrs=val, it is first decomposed into products of one-electron terms,
            e.g. a_p^ a_q^ a_r a_s = kDelta(q, r) * (a_p^ a_s) - (a_p^ a_r) * (a_q^ a_s).
        In case (p, r) is not an allowed transition(due to spin restrictions etc), a_r and a_s 
        can be exchanged with an extra minus sign (handled by `parity` in the code.)
        Finally, all two-electron terms can be decomposed into products of ALLOWED one-electron terms.
        
        The `mapping` is used to convert fermionic operators into qubit operators(typically Pauli terms).
        It is a dictionary whose keys are indices of allowed transitions (e.g. (p, q) if a_p^ a_q is allowed) 
        and values are the Pauli term corresponding to a_p^ a_q.
        
        Args:
            ferOp (qiskit.chemistry.FermionicOperator): second-quantized fermionic operator
            mapping (dict((tuple(int, int)), qiskit.quantum_info.Pauli)): 
                mapping from all allowed a_p^ a_q to Pauli operator

        """
        self.grouped_op = {}
        self.THRESHOLD = 1e-6
        self.mapping = fermionic2QubitMapping(n_so = ferOp.modes,
                                             n_e = num_electron, 
                                             labeling_method = labeling_method)
        
        h1, h2 = np.copy(ferOp.h1), np.copy(ferOp.h2)
        it1 = np.nditer(h1, flags=['multi_index'])
        it2 = np.nditer(h2, flags=['multi_index'])
        for h in it1:
            key = it1.multi_index
            self._add_an_h1(h, key)
        for h in it2:
            key = it2.multi_index
            self._add_an_h2(h, key)
        
    def _add_an_h1(self, coef, pq):
        """
            Add a single one-electron term into the grouped operator.  
            
            Args:
                coef (complex) : value of one-electron integral
                pq (tuple(int, int)): index of the one-electron term
        """
        if(abs(coef) < self.THRESHOLD): return 
        if pq in self.grouped_op.keys():
            self.grouped_op[pq] = self.grouped_op[pq] + coef
        else:
            self.grouped_op[pq] = coef
    
    def _add_an_h2(self, coef, pqrs):
        """
            Add a single two-electron term into the grouped operator. 
            
            Args:
                coef (complex) : value of two-electron integral
                pqrs (tuple(int, int, int, int)): index(in chemist notation) of the two-electron term
        """
        if(abs(coef) < self.THRESHOLD): return 
        parity = 1
        
        ## Note that in FermionicOperator, index (p,q,r,s) represents a_p^ a_r^ a_s a_q: chemist notation
        ## Here I use (p,q,r,s) to represent a_p^ a_q^ a_r a_s: physicist notation
        ## Thus the re-ordering of h-indices is needed
        p, s, q, r = pqrs

        ## Handle the exchange of a_r a_s if direct transformation will give illegal transitions
        if (((p, r) not in self.mapping.keys()) and ((r,p) not in self.mapping.keys())):
            r, s = s, r
            parity = -1
            print('Change to:', (p,q,r,s))
            
        ## a_p^ a_q^ a_r a_s = kDelta(q, r) * (a_p^ a_s) - (a_p^ a_r) * (a_q^ a_s)
        self._add_an_h1(parity * coef * kDelta(q, r), (p, s))
        mut_key = ((p, r), (q, s))
        if mut_key in self.grouped_op.keys():
            self.grouped_op[mut_key] -= coef * parity
        else:
            self.grouped_op[mut_key] = -coef * parity
            
    def to_paulis(self):
        """
        Convert the grouped fermionic operator into qubit operators (sum of Pauli terms)
        
        Returns:
            (qiskit.aqua.operators.WeightedPauliOperator) : qubit operator transformed based on `self.mapping`
        """
        
        mapping = self.mapping
        qubitOp = WeightedPauliOperator(paulis=[])
        for k, w in self.grouped_op.items():
            if np.ndim(k) == 1: ## one-e-term
                qubitOp += (w * mapping[k])
            elif np.ndim(k) == 2: ## 2-e-term
                k1, k2 = k
                qubitOp += (w * mapping[k1] * mapping[k2])
            else:
                raise ValueError('something wrong')
        qubitOp.chop(threshold=self.THRESHOLD)
        return qubitOp