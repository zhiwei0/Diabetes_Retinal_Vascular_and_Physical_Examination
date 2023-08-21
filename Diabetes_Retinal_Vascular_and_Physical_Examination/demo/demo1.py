from itertools import chain, combinations

def all_subsets(arr):
    return chain(*map(lambda x: combinations(arr, x), range(1, len(arr) + 1)))

def your_function(subset):
    # 在此处添加你的代码
    print(list(subset))

arr = [1, 2, 3]
for subset in all_subsets(arr):
    your_function(subset)
