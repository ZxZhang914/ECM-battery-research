import math

data = [2.246288 , 3.983960  , 1.441790  , 3.025670  , 0.512607  , 0.668136  ]

# Mean
mean_val = sum(data) / len(data)

# Population Std
variance = sum((x - mean_val)**2 for x in data) / len(data)
std_val = math.sqrt(variance)

print("Mean =", mean_val)
print("Std  =", std_val)
