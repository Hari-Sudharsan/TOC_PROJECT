import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

# REPLACE WITH YOUR ACTUAL API KEY
TOKEN = "YOUR_IBM_QUANTUM_TOKEN"

def run_hardware_validation(params, x_sample):
    service = QiskitRuntimeService(token=TOKEN, channel="ibm_quantum")
    # Select the least busy 7-qubit+ device
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=7)
    
    n = len(x_sample)
    qc = QuantumCircuit(n, 1)
    
    # Angle Encoding
    for i in range(n): qc.ry(x_sample[i], i)
    
    # Variational Layers (matching your training architecture)
    for l in range(params.shape[0]):
        for i in range(n):
            qc.ry(params[l,i,0], i)
            qc.rz(params[l,i,1], i)
        for i in range(n-1): qc.cx(i, i+1)
    
    qc.measure(0, 0)
    
    sampler = Sampler(backend=backend)
    job = sampler.run([qc], shots=1024)
    return job.result()

print("Hardware script ready. Input your IBM token to execute.")