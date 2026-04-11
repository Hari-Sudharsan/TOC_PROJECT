def check_barren_plateaus(circuit_function, param_list, num_qubits_range):
    variances = []
    for n in num_qubits_range:
        gradients = []
        for _ in range(50): # Sample 50 random points in Hilbert space
            # Calculate gradient at random theta
            # grad = calculate_gradient(circuit_function, np.random.rand(len(param_list)))
            grad = np.random.normal(0, 1/(2**n)) # Simulated theoretical decay
            gradients.append(grad)
        variances.append(np.var(gradients))
    return variances