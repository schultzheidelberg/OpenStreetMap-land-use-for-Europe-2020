from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from hsi_io import *
import numpy as np
import os
from hsi_io import loadtif, savetif
import time
import tqdm


img_list_FR = [
"until 032 finished",
"eu_countries_020",
"eu_countries_021",
"eu_countries_031",
"eu_countries_032",
"eu_countries_041",
"eu_countries_042",
"eu_countries_043",
"eu_countries_044",
"eu_countries_053",
"eu_countries_054",
"eu_countries_055",
"eu_countries_056",
"eu_countries_057",
"eu_countries_061",
"eu_countries_062",
"eu_countries_063",
"eu_countries_064",
"eu_countries_065",
"eu_countries_066",
"eu_countries_069",
"eu_countries_070",
"eu_countries_071",
"eu_countries_072",
"eu_countries_075",
"eu_countries_076",
"eu_countries_077",
"eu_countries_078",
"eu_countries_082",
"eu_countries_083",
"eu_countries_085",
"eu_countries_086",
"eu_countries_092",

 ]


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

img_list_RO = [
	"eu_countries_167",
	"eu_countries_168",
	"eu_countries_184",
	"eu_countries_185",
	"eu_countries_186",
	"eu_countries_202",
	"eu_countries_203",
	"eu_countries_204",
	"eu_countries_220",
	"eu_countries_221",
	"eu_countries_222",
	"eu_countries_235",
	"eu_countries_236",
	"eu_countries_237",
	"eu_countries_246",
 ]


img_list_AU = [

	"eu_countries_085",
	"eu_countries_095",
	"eu_countries_107",
	"eu_countries_108",
	"eu_countries_122",
	"eu_countries_123",
	"eu_countries_124",
	"eu_countries_138",
	"eu_countries_139",




 ]










def prediction(s2_image, classified_image):
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'AU_trained_model_single_10m.h5'

    ###########  Data Import #############
    # config the input and output file folder and location
    if extension(s2_image) == 'tif':
        bands_matrix = loadtif(s2_image)
        nbands_HSI = bands_matrix.shape[2]
        n_row = bands_matrix.shape[0]
        n_col = bands_matrix.shape[1]
        print(bands_matrix.shape)
        # check scaling to [-1;1]
        eps = 0.1
        if abs(np.max(bands_matrix) - np.min(bands_matrix)) > eps:
            bands_matrix_new = -1 + 2 * (bands_matrix - np.min(bands_matrix)) / (
                    np.max(bands_matrix) - np.min(bands_matrix))
        else:
            bands_matrix_new = bands_matrix
    bands_matrix_new_HSI = np.lib.pad(bands_matrix_new, ((6, 6), (6, 6), (0, 0)), 'symmetric')

    print('data is okay')
    ####################load the best model############################

    # perform feature extraction
    model_path = os.path.join(save_dir, model_name)
    model = load_model(model_path)
    feature_model = Model(inputs=model.input,
                          outputs=model.output)
    # initialize the deep feature map
    feature_map = np.empty([n_row, n_col])
    start = time.time()
    for i in range(n_row):
        X = range(0, n_col)
        X = [x + i * n_col for x in X]
        X_window_HSI = np.empty([n_col, 12, 12, nbands_HSI])
        for j in range(n_col):
            id = X[j]
            row_id = id // n_col
            col_id = id % n_col
            row_min = row_id
            row_max = row_id + 12
            col_min = col_id
            col_max = col_id + 12
            X_window_HSI[j, :, :, :] = bands_matrix_new_HSI[row_min:row_max, col_min:col_max, :]
        x_test_HSI = X_window_HSI.astype('float32')
        pred = feature_model.predict(x_test_HSI)
        pred = np.argmax(pred, axis=1)
        feature_map[i, :] = pred
        print('Classify the deep feature for row:', i)
    end = time.time()
    y_test_time = end - start

    #####################save the extracted features###########################


    feature_path = classified_image
    savetif(feature_map, s2_image, feature_path)
    print('Save classification labels into:%s ' % feature_path)
    print('prediction time', y_test_time)


def main():
    root = "/mnt/sds-hd/sd17f001/OSM4EO/sepal.io/"
    src_suf = "_10m/S2_10m_3035.tif"
    predicted_suf = "_10m/S2_10m_3035_classified.tif"

    for p in img_list_AU:
        src_tif = root + p + src_suf
        predicted_tif = root + p + predicted_suf
        #print(src_tif)
        #print(predicted_tif)
        prediction(src_tif, predicted_tif)

if __name__ == "__main__":
    main()
