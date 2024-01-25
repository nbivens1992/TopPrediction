import numpy as np


def ensure_power_of_2_length(sequence: np.ndarray, padding_value: int = -99) -> np.ndarray:
    current_rows = sequence.shape[0]

    # Calculate the number of rows needed to make it a power of 2
    if current_rows != 0:
        power_of_2 = 2 ** int(np.ceil(np.log2(current_rows)))
    else:
        power_of_2 = 1

    rows_needed_to_multiple_of_2 = power_of_2 - current_rows

    # Determine the total rows needed
    total_rows_needed = rows_needed_to_multiple_of_2

    if total_rows_needed > 0:
        # Create padding with the same number of columns as the sequence
        padding_shape = (total_rows_needed,) + sequence.shape[1:]
        padding = np.full(padding_shape, padding_value)

        # Append padding to the sequence
        sequence = np.vstack([sequence, padding])

    return sequence