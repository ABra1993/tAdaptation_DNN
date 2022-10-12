# %%

import numpy as np
import ujson as json
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# install required scripts
# from quickdraw_dataset.examples.binary_file_parser import *


def main():

    # define root
    linux = True # indicates whether script is run on mac or linux
    if linux:
        dir = '/home/amber/ownCloud/'
    else:
        dir = '/Users/a.m.brandsuva.nl/surfdrive/'

    # image classes
    classes = ['apple', 'car', 'face', 'fish', 'flower']


    path = dir+'Documents/code/DNN_adaptation/stimuli/full_simplified_' + classes[4] + '.ndjson'
    records = map(json.loads, open(path, encoding="utf8"))
    df = pd.DataFrame.from_records(records)

    print('Number of entries: ', len(df))
    print(df.columns)

    img = df.loc[0, 'drawing']

    strokes_n = len(img)

    values = np.ones((256, 256))

    for i in range(strokes_n):
        for j in range(len(img[i][0])):
            
            # x and y indices
            x = img[i][0][j]
            y = img[i][1][j]

            # select value
            values[x, y] = 0

    plt.imshow(values, cmap='gray')
    plt.show()

    # res =  cv2.resize(values, dsize=(28, 28))
    # plt.imshow(res, cmap='gray')
    # plt.show()


if __name__ == '__main__':
    main()