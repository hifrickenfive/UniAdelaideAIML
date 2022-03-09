def twoSum(nums: list, target: int) -> list:

    for num in nums:
        found = [number for number in nums if number == target-num]
        idx1 = nums.index(num)
        if len(found) > 1:
            idx2 = nums.index(found[1]) # this returns the first instance of the number :(
        else:
            idx2 = nums.index(found[0])
        if idx1 == idx2:
            continue
        else:
            return (idx1, idx2)


nums = [2,7,11,15]
target = 9

nums = [3, 3]
target = 6

result = twoSum(nums, target)
print(result)