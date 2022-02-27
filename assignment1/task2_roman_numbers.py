import math

def roman_numbers(roman_string: str) -> int:
    """Convert roman numerals to numbers.

    Args:
        roman_string (string) 

    Returns:
        number (int)
    """
    rome2num = {'I': 1, 'V': 5, 'X': 10,
                'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    sum = 0
    previous_num = math.inf

    for index, char in enumerate(roman_string):
        current_num = rome2num[char]
        if previous_num < current_num:
            sum -= 2*previous_num
            sum += current_num
        else:
            sum += current_num
            previous_num = current_num

    return sum

if __name__ == '__main__':
    # COMPSCI7327's test cases
    assert roman_numbers('III') == 3
    assert roman_numbers('LVIII') == 58
    assert roman_numbers('MCMXCIV') == 1994

    # Extra test cases
    assert roman_numbers('MMMCMXCIX') == 3999
    assert roman_numbers('DCXLIX') == 649
    assert roman_numbers('MMXXII') == 2022

    print('All test case passed.')