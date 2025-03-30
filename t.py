import random
import os

# Generate a random seed
for i in range(10):
    seed_value = int.from_bytes(os.urandom(4), 'big')
    random.seed(seed_value)
    print(seed_value)
