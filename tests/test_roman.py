import roman_numbers # The code to test

def test_basic1():
    assert roman_numbers.convert_to_numbers('III') == 3

def test_basic2():
    assert roman_numbers.convert_to_numbers('LVIII') == 58

def test_basic3():
    assert roman_numbers.convert_to_numbers('MCMXCIV') == 1994

def test_hard1():
    assert roman_numbers.convert_to_numbers('MMMCMXCIX') == 3999

def test_hard2():
    assert roman_numbers.convert_to_numbers('DCXLIX') == 649