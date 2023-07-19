import os
import sys
import json
path = "/mnt/home/irielab/workspace/projects/imageTransactionTest_2/class_maps/class_map.json"

def main():
    with open(path, 'r') as f:
        j = json.load(f)
    
    for k in j.keys():
        cnt = 0
        for pred in j[k]:
            if str(pred) == k:
                cnt += 1
        acc = cnt*100 / len(j[k])
        print(f"target:{k}, each Acc: {acc}%")


if __name__ == '__main__':
    main()