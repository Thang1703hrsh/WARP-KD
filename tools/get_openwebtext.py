import datasets
import os
import re

dataset = datasets.load_dataset('Elriggs/openwebtext-100k', split='train', trust_remote_code=True)

os.makedirs("data/openwebtext", exist_ok=True)

num = 0
with open("data/openwebtext/data.txt", "w") as f:
    for data in dataset:
        f.write(re.sub(r"\n+", "<@x(x!>", data['text']) + "\n")
        num += 1
        if num >= 250000:
            break

print("Number of lines:", num)