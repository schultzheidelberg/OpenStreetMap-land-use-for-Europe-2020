# Author: Zhaoyan Wu
import os
import sys
import numpy as np
import rasterio
from rasterio.features import sieve, shapes

img_list = [
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

def seieve(in_file, out_file, size):
    with rasterio.Env():
        with rasterio.open(in_file) as src:
            shade = src.read(1)

        sieved = sieve(shade, size, out=np.zeros(src.shape, src.dtypes[0]), connectivity=8)

        kwargs = src.meta
        kwargs['transform'] = rasterio.transform.guard_transform(kwargs['transform'])
        with rasterio.open(out_file, 'w', **kwargs) as dst:
            dst.write(sieved, indexes=1)

def main():
    # if len(sys.argv) > 1:
    #     size = int(sys.argv[1])
    # else:
    #     print("Usage: python post_sieve_filtering.py win_size")
    #     exit(0)
    size = 64
    folder_path = '/mnt/sds-hd/sd17f001/OSM4EO/sepal.io'
    in_name = 'S2_10m_3035_classified.tif'
    out_name = 'S2_10m_3035_classified_sieved.tif'
    for img in img_list:
        print(img)
        in_file = os.path.join(folder_path, img+'_10m', in_name)
        out_file = os.path.join(folder_path, img+'_10m', out_name)
        seieve(in_file, out_file, size)


if __name__ == '__main__':
    main()
