file_path = "annotations.txt"

with open(file_path, 'r') as f:
    lines = [line.strip().split("\t")[2] for line in f.readlines()]

with open("target_annotations.txt", "w") as f:
    for line in lines:
        f.write(line + "\n")
        