def count_carries(number_a: int, number_b: int) -> int:
    """Count the carries.

    Assumptions:
    1. Addition only.
    2. Negative numbers excluded.
    """
    num_carries = 0
    number_c = number_a + number_b

    digits_a = [int(a) for a in str(abs(number_a))]
    digits_b = [int(b) for b in str(abs(number_b))]
    digits_c = [int(c) for c in str(abs(number_c))]

    digits_a.reverse()
    digits_b.reverse()
    digits_c.reverse()

    num_least_digits = min(len(digits_a), len(digits_b))

    for i in range(num_least_digits):
        if digits_c[i] < digits_b[i] + digits_a[i]:
            num_carries += 1

    return num_carries