# row anchors are a series of pre-defined coordinates in image height to detect lanes
# the row anchors are defined according to the evaluation protocol of CULane and Tusimple
# since our method will resize the image to 288x800 for training, the row anchors are defined with the height of 288
# you can modify these row anchors according to your training image resolution

tusimple_row_anchor = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112,
                       116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
                       168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
                       220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
                       272, 276, 280, 284]
culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]

new_tusimple_anchor = [85, 90, 96, 101, 106, 112, 117, 122, 128, 133, 138, 144, 149, 154, 160, 165, 170, 176, 181, 186,
                       192, 197, 202, 208, 213, 218, 224, 229, 234, 240, 245, 250, 256, 261, 266, 272, 277, 282, 288,
                       293,
                       298, 304, 309, 314, 320, 325, 330, 336, 341, 346, 352, 357, 362, 368, 373, 378]

new_culane_anchor = [161, 174, 188, 200, 213, 226, 240, 252, 265, 278, 292, 304, 317, 330, 344, 356, 369, 382]

if __name__ == '__main__':
    new_anchor = [int(float(i) * 384. / 288.) for i in culane_row_anchor]
    print(new_anchor)
