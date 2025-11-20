import math

data = [4.861450, 7.123941 ,  2.300177  , 3.832844   , 3.107630, 0.388822]

# Mean
mean_val = sum(data) / len(data)

# Population Std
variance = sum((x - mean_val)**2 for x in data) / len(data)
std_val = math.sqrt(variance)

print("Mean =", mean_val)
print("Std  =", std_val)
