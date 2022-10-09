"""
Combines image and label files in different directories into a shared address
"""
import os
import shutil

folder = '/Users/aungriah/Desktop/Finetuning'
bags = ['zurich_bag', 'SABAG1', 'SABAG2']
labels = ['zurich_labels', 'sabag1_label', 'sabag2_label']

os.makedirs(os.path.join(folder, 'allImages'), exist_ok=True)
os.makedirs(os.path.join(folder, 'allLabels'), exist_ok=True)

counter = 0

for idx in range(len(bags)):
    images = os.listdir(os.path.join(folder, bags[idx]))
    images.sort()
    bag_labels = os.listdir(os.path.join(folder, labels[idx]))
    bag_labels.sort()
    print(len(images), len(bag_labels))
    assert len(images) == len(bag_labels), 'Length of {} and {} are not equal!'.format(bags[idx],labels[idx])

    for sub_idx in range(len(images)):
        assert images[sub_idx][:-4] == bag_labels[sub_idx][:-4]

        shutil.copy(os.path.join(folder, bags[idx], images[sub_idx]), os.path.join(folder, 'allImages') + '/image_{}.{}'.format(counter, images[sub_idx][-3:]))
        shutil.copy(os.path.join(folder, labels[idx], bag_labels[sub_idx]),
                    os.path.join(folder, 'allLabels') + '/image_{}.xml'.format(counter))

        counter +=1