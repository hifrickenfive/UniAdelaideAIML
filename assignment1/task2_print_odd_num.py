def print_odd_num(numbers):
    odd_numbers = [num for num in numbers if num % 2 > 0]
    return odd_numbers

if __name__ == '__main__':
    # COMPSCI7327's test cases
    assert print_odd_num([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) ==  [1, 3, 5, 7, 9, 11, 13, 15]

    # Extra test case
    assert print_odd_num([1, 1, 0, -1]) == [1, 1, -1]
    print('All test case passed.')