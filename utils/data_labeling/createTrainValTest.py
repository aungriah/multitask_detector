"""
Split training into training, validation and test
"""

import os, random
labels = '/Users/aungriah/Desktop/Finetuning/finetuning/label'
random.seed(23)

list_lanes = []
list_obj = []
for line in open('/Users/aungriah/Desktop/Finetuning/finetuning/train_gt.txt', 'r'):
    #line = line.rstrip()
    obj_file = line.split()[0][:-3] + 'txt'
    list_obj.append(obj_file)
    list_lanes.append(line)

nr_pics = len(list_lanes)

combined = list(zip(list_obj, list_lanes))

random.shuffle(combined)

obj, lanes = zip(*combined)
obj, lanes = [a for a in obj], [b for b in lanes]
train_obj = obj[:int(0.5*nr_pics)]
train_obj.sort()
train_lanes = lanes[:int(0.5*nr_pics)]
train_lanes.sort()

val_obj = obj[int(0.5*nr_pics):int(0.75*nr_pics)]
val_obj.sort()
val_lanes = lanes[int(0.5*nr_pics):int(0.75*nr_pics)]
val_lanes.sort()
test_obj = obj[int(0.75*nr_pics):]
test_obj.sort()
test_lanes = lanes[int(0.75*nr_pics):]
test_lanes.sort()

with open('/Users/aungriah/Desktop/Finetuning/finetuning/train_obj.txt', 'w') as f:
    for elem in train_obj:
        f.writelines(elem + '\n')
f.close()
with open('/Users/aungriah/Desktop/Finetuning/finetuning/val_obj.txt', 'w') as f:
    for elem in val_obj:
        f.writelines(elem + '\n')
f.close()
with open('/Users/aungriah/Desktop/Finetuning/finetuning/train_lanes.txt', 'w') as f:
    for elem in train_lanes:
        f.writelines(elem)
f.close()
with open('/Users/aungriah/Desktop/Finetuning/finetuning/val_lanes.txt', 'w') as f:
    for elem in val_lanes:
        f.writelines(elem)
f.close()
with open('/Users/aungriah/Desktop/Finetuning/finetuning/test_obj.txt', 'w') as f:
    for elem in test_obj:
        f.writelines(elem + '\n' )
f.close()
with open('/Users/aungriah/Desktop/Finetuning/finetuning/test_lanes.txt', 'w') as f:
    for elem in test_lanes:
        f.writelines(elem)
f.close()

