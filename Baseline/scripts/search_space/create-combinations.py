from itertools import combinations
import json

transformations = [
    "randomcrop224",
    "morpho_erosion",
    "morpho_dilation",
    "affine",
    "colorjitter",
    "hflip",
    "invert",
    "gaussianblur",
    "gray"
]




# Generate second-order combinations ensuring 'resize256' appears only once
second_order_combinations = ["randomcrop224," + c for c in transformations[1:]]
print(f"Total number of second order combinations: {len(second_order_combinations)}")


# Generate third-order combinations ensuring 'resize256' appears only once
third_order_combinations = ["randomcrop224," + ",".join(c) for c in combinations(transformations[1:], 2)]
print(f"Total number of third_order_combinations: {len(third_order_combinations)}")

# Generate fourth-order combinations ensuring 'resize256' appears only once
fourth_order_combinations = ["randomcrop224," + ",".join(c) for c in combinations(transformations[1:], 3)]
print(f"Total number of fourth_order_combinations: {len(fourth_order_combinations)}")

# Combine all combinations
all_combinations = ["randomcrop224"] + second_order_combinations + third_order_combinations + fourth_order_combinations

# # Print all combinations
# for combination in all_combinations:
#     print(combination)


transforms = {"all": all_combinations}


print(f"Total number of combinations: {len(all_combinations)}")
with open("all_permuations.json", 'w') as f:
    json.dump(transforms, f, indent=4)

# If you need to return as a list, you can simply return the list:
# return all_combinations
