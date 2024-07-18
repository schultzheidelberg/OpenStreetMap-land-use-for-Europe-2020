
import json
import rasterio
from rasterio.mask import mask


def clip_geojson(in_tif, geometry, out_tif):

    with open(geometry) as src_geo:
        geoms = json.loads(src_geo.read())
        geoms = geoms['features'][0]['geometry']
    shape = [geoms]

    with rasterio.open(in_tif) as src_ds:
        out_img, out_transform = mask(src_ds, shape, crop=True)
        kwds = src_ds.meta
    print('clipping is now successful!')
    # update the meta data
    kwds.update({'driver': 'GTiff',
                 'height': out_img.shape[1],
                 'width': out_img.shape[2],
                 'transform': out_transform
                 })
    with rasterio.open(out_tif, "w", **kwds) as dst_ds:
        dst_ds.write(out_img)


if __name__ == "__main__":
    root = "Z:/OSM4EO/val_dataset/"
    file = "DES2_10m_3035_classified_sieved_raw.tif"
    saved_root = "Z:/OSM4EO/val_dataset/"
    saved_file = "DE_10m_3035_classified_sieved_raw.tif"
    geo_file = "C:/Users/haoli/Desktop/OSM4EO/DE_boundary.geojson"
    img_input = root + file
    img_saved = saved_root + saved_file
    clip_geojson(img_input, geo_file, img_saved)
