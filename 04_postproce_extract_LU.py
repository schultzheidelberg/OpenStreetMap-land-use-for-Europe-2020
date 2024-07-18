import json
import os
import numpy as np
import rasterio
from tqdm import tqdm


def loadtif(filename):
    # Open a single band and plot
    with rasterio.open(filename) as src:
        a = src.read()
        nbands = src.count
        nrows = src.height
        ncolumns = src.width
    image = np.empty([nrows, ncolumns, nbands])
    for i in range(nbands):
        image[:, :, i] = a[i, :, :]
    return image


def savetif(X, infilename, outfilename):
    # save classification map into a same georeferening tif
    with rasterio.open(infilename) as src:
        kwds = src.profile
        kwds['count'] = 1
    X = X.astype(np.uint16)
    with rasterio.open(outfilename, 'w', **kwds) as dst:
        dst.write(X, 1)


def extract_LU_layers(img_input, img_saved, LU_class):
    classified_dataset = rasterio.open(img_input)
    classified_label = np.array(classified_dataset.read(1))
    filtered_label = np.where(LU_class == classified_label, LU_class, 0)
    savetif(filtered_label, img_input, img_saved)
    print('Save the classified layers into:%s ' % img_saved)


    pass


if __name__ == "__main__":
    # please define the LULC classes that you what to extract.
    LU_class = 12
    # this is the image list you what to extract.
    img_list_DE = [
     "077",
     "078",
     "079",
     "080",
     "085",
     "086",
     "087",
     "088",
     "089",
     "095",
     "097",
     "098",
     "108",
     "110",
     "111",
     "112",
     "109",
     "123",
     "124",
     "125",
     "126",
     "096",

    ]
    for p in img_list_DE:
        # define the root repo, input file, and saved file names.
        root = "C:/Users/haoli/Desktop/OSM4EO/DE/"
        saved_root = "C:/Users/haoli/Desktop/OSM4EO/DE/LU_layers/"
        if not os.path.exists(saved_root):
            os.mkdir(saved_root)
            print("Directory ", saved_root, " Created ")
        file = "_S2_10m_3035_classified.tif"
        img_input = root + p + file
        img_saved = saved_root + p + "_LU" + str(LU_class) + file
        extract_LU_layers(img_input, img_saved, LU_class)
