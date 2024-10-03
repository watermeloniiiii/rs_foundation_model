import os


def find_last_index(name, dir):
    idx_lst = []
    for model in os.listdir(dir):
        if name in model:
            idx_lst.append(int(model.split("_")[-1]))
    if idx_lst == []:
        return 1
    return max(idx_lst) + 1
