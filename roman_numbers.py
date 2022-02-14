import math

def convert_to_numbers(roman_string):
    rome2num = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M':1000}
    sum = 0
    previous_num = math.inf

    for index, char in enumerate(roman_string):
        current_num = rome2num(char)
        if previous_num < current_num:
            sum -= 2*previous_num 
            sum += current_num
        else:
            sum += current_num
            previous_num = current_num

    return sum