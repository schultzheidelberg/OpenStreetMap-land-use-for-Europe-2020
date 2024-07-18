import json
import os
import numpy as np
import rasterio
from tqdm import tqdm
import json
import os
import numpy as np
import rasterio

transform_map = {
    41: 11,
    21: 5,
    5: 12,
    11: 1,
    12: 2,
    13: 3,
    14: 4,
    22: 6,
    23: 7,
    31: 8,
    32: 9,
    33: 10,
}


def transform_label(in_tif, out_tif):
    """transform michi's label to one-hot label

    :param in_tif: input tif path
    :param out_tif: transformed label tif output path
    """
    with rasterio.open(in_tif) as src_ds:
        kwds = src_ds.profile
        label = src_ds.read(1)

        for item in transform_map.items():
            label[label == item[1]] = item[0]

        with rasterio.open(out_tif, "w", **kwds) as dst_ds:
            dst_ds.write(label, 1)


if __name__ == "__main__":
    # this is the image list you what to mask.
    img_list_DE = [
        "eu_countries_077",
        "eu_countries_078",
        "eu_countries_079",
        "eu_countries_080",
        "eu_countries_085",
        "eu_countries_086",
        "eu_countries_087",
        "eu_countries_088",
        "eu_countries_089",
        "eu_countries_095",
        "eu_countries_097",
        "eu_countries_098",
        "eu_countries_099",
        "eu_countries_108",
        "eu_countries_109",
        "eu_countries_110",
        "eu_countries_111",
        "eu_countries_112",
        "eu_countries_123",
        "eu_countries_124",
        "eu_countries_125",
        "eu_countries_126",
        "eu_countries_096",
    ]
    for p in img_list_DE:
        # define the root repo, input file, and saved file names.
        folder_path = '/mnt/sds-hd/sd17f001/OSM4EO/sepal.io'
        in_name = 'S2_10m_3035_classified_clipped.tif'
        out_name = 'S2_10m_3035_classified_transferred.tif'
        img_input = os.path.join(folder_path, p + '_10m', in_name)
        img_saved = os.path.join(folder_path, p + '_10m', out_name)
        transform_label(img_input, img_saved)
