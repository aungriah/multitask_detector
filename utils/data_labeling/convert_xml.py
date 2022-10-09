"""
Convert xml annotation files into TuSimple format
"""
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import json

h_samples_orig = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
h_samples = np.asarray([160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710])
h_samples = np.int32(np.asarray(h_samples))

path_to_xmls = '/Users/aungriah/Documents/eval/labels'
path_to_images = '/Users/aungriah/Documents/eval/images'


path_to_json = '/Users/aungriah/Documents/eval'
path_to_txt = '/Users/aungriah/Documents/eval/bboxes'
os.makedirs(path_to_txt, exist_ok=True)

files = os.listdir(path_to_xmls)
files.sort()

dict_list = []

with open(path_to_json + '/data.txt', 'w') as f:
    for file in files:
        tree = ET.parse(path_to_xmls + '/'+file)
        root = tree.getroot()
        filename = root.find('filename').text

        labels = []
        bboxes = []
        vertices = []
        polygons = []

        for boxes in root.iter('object'):


            type = boxes.find('name').text
            if type == 'lane':
                vertices_indiv = []
                for coord in boxes.iter('line'):
                    for i in range(len(coord)):
                        vertices_indiv.append(float(coord[i].text))
                vertices.append(np.asarray(vertices_indiv))
                continue

            ymin = float(boxes.find("bndbox/ymin").text)
            xmin = float(boxes.find("bndbox/xmin").text)
            ymax = float(boxes.find("bndbox/ymax").text)
            xmax = float(boxes.find("bndbox/xmax").text)
            labels.append(type)
            bboxes.append([xmin, ymin, xmax, ymax])



        # Write bounding boxes to file
        with open(path_to_txt + '/'+filename[:-4] + '.txt', 'w') as fo:
            for bbox, label in zip(bboxes, labels):
                x1,y1,x2,y2 = [str(elem) for elem in bbox]
                line = str(label) + ' ' + x1 + ' ' + y1 + ' ' + x2 + ' ' + y2 + '\n'
                fo.writelines(line)

        fo.close()

        img = cv2.imread(path_to_images + '/' + filename, cv2.COLOR_BGR2RGB)
        for box in bboxes:
            x1,y1,x2,y2 = [int(elem) for elem in box]
            img = cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)

        dict = {}
        x_samples = []
        for line in vertices:
            print(file)
            line = line.reshape(-1,2)
            print(line)
            xcor = line[:,0]
            ycor = 720-line[:,1]
            ymin = np.min(line[:,1])
            ymax = np.max(line[:,1])
            poly = np.polyfit(ycor, xcor,3)
            xvalues = []

            for i, sample in enumerate(h_samples):
                if ymin<=sample<=ymax:
                    x = np.polyval(poly,(720-1-sample))
                else:
                    x = -2
                xvalues.append(x)
            x_samples.append(xvalues)

            for i,cor in enumerate(xvalues):
                if cor == -2:
                    continue
                pp = (int(cor), int(h_samples[i]))
                img = cv2.circle(img, pp, 5, (0, 255, 0), -1)
            cv2.imwrite('/Users/aungriah/Documents/eval/check/' + filename, img)

        dict["lanes"] = x_samples
        dict["h_samples"] = h_samples_orig
        dict["raw_file"] = os.path.join('eval', 'images',filename)

        f.writelines(str(json.dumps(dict)) + '\n')

f.close()

# with open(path_to_json + '/data.json', 'w') as outfile:
#     json.dump(dict_list, outfile)










    # for xvals in x_samples:
    #     for i,cor in enumerate(xvals):
    #         if cor == -2:
    #             continue
    #         pp = (int(cor), int(h_samples[i]))
    #         img = cv2.circle(img, pp, 5, (0, 255, 0), -1)

    # print('/Users/aungriah/Desktop/check/'+ file[:-4]+'.png')
    # cv2.imwrite('/Users/aungriah/Desktop/check/'+ file[:-4]+'.png', img)

    # f = plt.figure()
    # ax = f.add_subplot(1,1,1)
    # plt.imshow(img)
    # plt.show()




#     line = line.reshape(-1,1,2)
#     isClosed = False
#     color = (255, 0, 0)
#     thickness = 2
#     img = cv2.polylines(img, np.int32([line]), isClosed, color, thickness)
#
#
# f = plt.figure()
# ax = f.add_subplot(1,1,1)
# plt.imshow(img)
#
# for bbox in bboxes:
#     x1, y1, x2, y2 = bbox
#     rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='g', facecolor='none')
#     ax.add_patch(rect)
# plt.show()






