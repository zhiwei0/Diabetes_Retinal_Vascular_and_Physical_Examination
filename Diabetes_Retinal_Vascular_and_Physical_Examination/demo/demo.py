def forward_search(arr, your_function, subset=None):
    if subset is None:
        subset = []

    # 调用你的函数
    your_function(subset)

    # 终止条件：如果子集的大小等于数组的大小，则返回
    if len(subset) == len(arr):
        return

    # 前向搜索：逐个添加元素，并递归调用自身
    for i in range(len(arr)):
        if arr[i] not in subset:
            new_subset = subset.copy()
            new_subset.append(arr[i])
            forward_search(arr, your_function, new_subset)

def your_function(subset):
    # 在此处添加你的代码
    print(subset)

arr = ['artery_caliber', 'vein_caliber', 'frac', 'AVR']
    # , 'artery_curvature', 'vein_curvature', 'artery_BSTD', 'vein_BSTD', 'artery_simple_curvatures', 'vein_simple_curvatures', 'artery_Branching_Coefficient', 'vein_Branching_Coefficient', 'artery_Num1stBa', 'vein_Num1stBa', 'artery_Branching_Angle', 'vein_Branching_Angle', 'artery_Angle_Asymmetry', 'vein_Angle_Asymmetry', 'artery_Asymmetry_Raito', 'vein_Asymmetry_Raito', 'artery_JED', 'vein_JED', 'artery_Optimal_Deviation', 'vein_Optimal_Deviation', 'vessel_length_density', 'vessel_area_density']

forward_search(arr, your_function)
