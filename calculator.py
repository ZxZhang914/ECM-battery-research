import math

data = [1.175127, 6.563030, 4.450194, 5.456630]

# Mean
mean_val = sum(data) / len(data)

# Population Std
variance = sum((x - mean_val)**2 for x in data) / len(data)
std_val = math.sqrt(variance)

print("Mean =", mean_val)
print("Std  =", std_val)
