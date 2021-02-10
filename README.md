# qubit-efficient-mapping

## Introduction
A qubit-efficient tool to map second-quantized fermionic operators into qubit(Pauli) operators for further variational quantum algorithms. 

## Dependencies
- Qiskit: 0.23.5

## Tutorial 
First, construct a `QMolecule` object with `PySCFDriver`, e.g. a hydrogen molecule with interatomic distance `dist`(A) in STO-3G basis set. 
```python
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
dist = .7
driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 " + str(dist), unit=UnitsType.ANGSTROM, 
                         charge=0, spin=0, basis='sto-3g')
molecule = driver.run()
num_electrons = molecule.num_alpha + molecule.num_beta
```

Then, a `FermionicOperator` object can be constructed using one- and two-body intergrals in `molecule`.

```python
ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
```
Now, `groupedFermionicOperator` helps us tranform the fermionic operators(`ferOp`) into qubit operators(`qubitOp`). 
The `to_paulis` method returns the qubit operator based on our mapping method. 
```python
from groupedFermionicOperator import groupedFermionicOperator

g = groupedFermionicOperator(ferOp, num_electrons)
qubitOp = g.to_paulis()
```

To verify the result, one may calculate the lowest eigenvalue of the qubit operator, 
and compare the result with that from built-in Jordan-Wigner transform etc.
```python
from qiskit.aqua.algorithms import NumPyEigensolver
result_exact = NumPyEigensolver(qubitOp).run()
energy_exact = (np.real(result_exact.eigenvalues))[0]
print(energy_exact)

qubitOpJW = ferOp.mapping(map_type='jordan_wigner', threshold=1e-6)
result_exactJW = NumPyEigensolver(qubitOpJW).run()
energy_exactJW = (np.real(result_exactJW.eigenvalues))[0]
print(energy_exactJW)
```
