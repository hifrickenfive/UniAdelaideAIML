import count_carries # The code to test

def test_sample_cases1():
    assert count_carries.count_carries(123, 456) == 0

def test_sample_cases2():
    assert count_carries.count_carries(555, 555) == 3

def test_sample_cases3():
    assert count_carries.count_carries(123, 594) == 1

def test_corner_case1():
    assert count_carries.count_carries(900, 100) == 1

def test_minus():
    assert count_carries.count_carries(900, -100) == 0