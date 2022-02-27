import print_odd_num

def test_basic1():
    assert print_odd_num.print_odd_num([1, 2, 3, 4]) == [1, 3]

def test_basic2():
    assert print_odd_num.print_odd_num([999, 3, 1, 2]) == [999, 3, 1]

def test_basic3():
    assert print_odd_num.print_odd_num([1]) == [1]

def test_basic4():
    assert print_odd_num.print_odd_num([2]) == []