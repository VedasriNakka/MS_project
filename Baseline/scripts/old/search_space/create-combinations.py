from itertools import combinations

transformations = [
    "resize256",
    "morpho_erosion",
    "morpho_dilation",
    "affine",
    "colorjitter",
    "hflip",
    "invert",
    "gaussianblur"
]

# Generate second-order combinations ensuring 'resize256' appears only once
second_order_combinations = ["resize256," + c for c in transformations[1:]]

# Generate third-order combinations ensuring 'resize256' appears only once
third_order_combinations = ["resize256," + ",".join(c) for c in combinations(transformations[1:], 2)]

# Generate fourth-order combinations ensuring 'resize256' appears only once
fourth_order_combinations = ["resize256," + ",".join(c) for c in combinations(transformations[1:], 3)]

# Combine all combinations
all_combinations = ["resize256"] + second_order_combinations + third_order_combinations + fourth_order_combinations

# Print all combinations
for combination in all_combinations:
    print(combination)

# If you need to return as a list, you can simply return the list:
# return all_combinations
