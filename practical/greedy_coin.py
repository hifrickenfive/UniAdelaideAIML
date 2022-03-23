ONECENT = 1
FIVECENT = 5
TENCENT = 10
TWENTYFIVECENT = 25

def main(cur_money):
    num_25 = 0
    num_10 = 0
    num_5 = 0
    num_1 = 0

    while cur_money >= TWENTYFIVECENT:
        num_25 += 1
        cur_money -= TWENTYFIVECENT
    
    while cur_money >= TENCENT:
        num_10 += 1
        cur_money -= TENCENT
    
    while cur_money >= FIVECENT:
        num_5 += 1
        cur_money -= FIVECENT
    
    while cur_money >= ONECENT:
        num_1 += 1
        cur_money -= ONECENT
    
    return [num_25, num_10, num_5, num_1]

if __name__ == '__main__':
    result = main(41)
    print(result)