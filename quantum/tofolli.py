import numpy as np


def toffoli_gate():
    """Standard Toffoli gate matrix"""
    toffoli = np.eye(8, dtype=complex)
    toffoli[6, 6] = 0
    toffoli[6, 7] = 1
    toffoli[7, 6] = 1
    toffoli[7, 7] = 0
    return toffoli


def test_gate_implementation(gate_name, inputs, expected_outputs, circuit_func):
    """Helper function to test gate implementations"""
    print(f"\n{gate_name} Implementation:")
    print("Input → Output (Expected)")
    print("-" * 25)

    for inp, expected in zip(inputs, expected_outputs):
        result = circuit_func(*inp)
        status = "✓" if result == expected else "✗"
        print(f"{inp} → {result} ({expected}) {status}")


def toffoli_as_not(a):
    """
    NOT gate using Toffoli: Set both controls to |1⟩
    Toffoli(1,1,a) = (1,1,NOT(a))
    """
    # Apply Toffoli with controls set to 1
    state_index = 1 * 4 + 1 * 2 + a  # |11a⟩
    input_state = np.zeros(8, dtype=complex)
    input_state[state_index] = 1

    output_state = toffoli_gate() @ input_state
    output_index = np.argmax(np.abs(output_state))

    # Extract the target bit (least significant bit)
    return output_index & 1


def toffoli_as_and(a, b):
    """
    AND gate using Toffoli: Use ancilla bit set to |0⟩
    Toffoli(a,b,0) = (a,b,AND(a,b))
    """
    state_index = a * 4 + b * 2 + 0  # |ab0⟩
    input_state = np.zeros(8, dtype=complex)
    input_state[state_index] = 1

    output_state = toffoli_gate() @ input_state
    output_index = np.argmax(np.abs(output_state))

    return output_index & 1


def toffoli_as_nand(a, b):
    """
    NAND gate using Toffoli: Use ancilla bit set to |1⟩
    Toffoli(a,b,1) = (a,b,NAND(a,b))
    """
    state_index = a * 4 + b * 2 + 1  # |ab1⟩
    input_state = np.zeros(8, dtype=complex)
    input_state[state_index] = 1

    output_state = toffoli_gate() @ input_state
    output_index = np.argmax(np.abs(output_state))

    return output_index & 1


def toffoli_as_or(a, b):
    """
    OR gate using Toffoli + NOT: OR(a,b) = NOT(NAND(NOT(a), NOT(b)))
    This requires multiple Toffoli gates in practice
    """
    # For demonstration, we'll compute it classically
    # In practice: NOT(a) → NAND(NOT(a), NOT(b)) → NOT of that result
    not_a = toffoli_as_not(a)
    not_b = toffoli_as_not(b)
    nand_result = toffoli_as_nand(not_a, not_b)
    return toffoli_as_not(nand_result)


def toffoli_as_nor(a, b):
    """
    NOR gate: NOR(a,b) = NOT(OR(a,b))
    """
    or_result = toffoli_as_or(a, b)
    return toffoli_as_not(or_result)


def toffoli_as_xor(a, b):
    """
    XOR gate using Toffoli: XOR(a,b) = OR(AND(a,NOT(b)), AND(NOT(a),b))
    Requires multiple Toffoli gates
    """
    # XOR(a,b) = (a AND NOT(b)) OR (NOT(a) AND b)
    not_a = toffoli_as_not(a)
    not_b = toffoli_as_not(b)

    term1 = toffoli_as_and(a, not_b)
    term2 = toffoli_as_and(not_a, b)

    return toffoli_as_or(term1, term2)


def toffoli_as_fanout(a):
    """
    FANOUT (copying): Copy bit a to two outputs
    Toffoli(a,1,0) = (a,1,a)  # Copies a to third bit
    """
    state_index = a * 4 + 1 * 2 + 0  # |a10⟩
    input_state = np.zeros(8, dtype=complex)
    input_state[state_index] = 1

    output_state = toffoli_gate() @ input_state
    output_index = np.argmax(np.abs(output_state))

    # Return both copies: (original_a, copy_of_a)
    bit0 = (output_index >> 2) & 1  # First bit
    bit2 = output_index & 1  # Third bit
    return (bit0, bit2)


def demonstrate_universal_computation():
    """
    Show that Toffoli can implement any Boolean function
    Example: Full adder sum and carry bits
    """
    print("\nFull Adder using Toffoli gates:")
    print("Inputs: a, b, c_in → Outputs: sum, c_out")
    print("-" * 40)

    for a in [0, 1]:
        for b in [0, 1]:
            for c_in in [0, 1]:
                # Sum = XOR(XOR(a,b), c_in)
                xor_ab = toffoli_as_xor(a, b)
                sum_bit = toffoli_as_xor(xor_ab, c_in)

                # Carry = OR(AND(a,b), AND(XOR(a,b), c_in))
                and_ab = toffoli_as_and(a, b)
                and_xor_cin = toffoli_as_and(xor_ab, c_in)
                carry_out = toffoli_as_or(and_ab, and_xor_cin)

                print(f"({a},{b},{c_in}) → sum={sum_bit}, carry={carry_out}")


def show_complexity_analysis():
    """
    Show the gate complexity for different operations
    """
    print("\nGate Complexity (number of Toffoli gates needed):")
    print("-" * 50)
    complexity = {
        "NOT": 1,
        "AND": 1,
        "NAND": 1,
        "OR": 3,  # NOT + NAND + NOT
        "NOR": 4,  # NOT + NAND + NOT + NOT
        "XOR": 5,  # Multiple AND, NOT, OR operations
        "FANOUT": 1,
        "Full Adder": "~13",  # Approximate for sum + carry
    }

    for gate, count in complexity.items():
        print(f"{gate:12}: {count} Toffoli gate(s)")


def quantum_vs_classical_completeness():
    """
    Explain the difference between classical and quantum universality
    """
    print("\nUniversality Comparison:")
    print("-" * 25)
    print("Classical Reversible Computing:")
    print("  • Toffoli alone is universal")
    print("  • Can implement any Boolean function")
    print("  • Preserves information (reversible)")

    print("\nQuantum Computing:")
    print("  • Toffoli + Hadamard = quantum universal")
    print("  • Toffoli handles classical logic")
    print("  • Hadamard creates superposition")
    print("  • Need both for full quantum computation")


if __name__ == "__main__":
    # Test basic gate implementations
    binary_inputs = [(0,), (1,)]
    test_gate_implementation(
        "NOT Gate", binary_inputs, [1, 0], toffoli_as_not  # Expected NOT outputs
    )

    binary_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    test_gate_implementation(
        "AND Gate", binary_pairs, [0, 0, 0, 1], toffoli_as_and  # Expected AND outputs
    )

    test_gate_implementation(
        "NAND Gate",
        binary_pairs,
        [1, 1, 1, 0],  # Expected NAND outputs
        toffoli_as_nand,
    )

    test_gate_implementation(
        "OR Gate", binary_pairs, [0, 1, 1, 1], toffoli_as_or  # Expected OR outputs
    )

    test_gate_implementation(
        "XOR Gate", binary_pairs, [0, 1, 1, 0], toffoli_as_xor  # Expected XOR outputs
    )

    # Test FANOUT
    print("\nFANOUT Implementation:")
    for a in [0, 1]:
        result = toffoli_as_fanout(a)
        print(f"FANOUT({a}) → {result}")

    # Demonstrate complex computation
    demonstrate_universal_computation()

    # Show complexity analysis
    show_complexity_analysis()

    # Compare universality
    quantum_vs_classical_completeness()
