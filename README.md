### OSM4EO LULC classficaiton with deep residual convolutional neural networks

existing models: https://1drv.ms/f/s!AsaD57Gg8TWIlfFvgtTjDT6E0067Jg?e=kkmekb 

This is the source code repo for the deep learning method within OSM4EO work.

To be noticed, all the code was designed for bwcluster GPU server with python packages and environment on Anaconda. 

**Importantly, all trained models for different EU countries can be found in saved_models repo.**

### Current results collection for EU countries


| Country | Classification | Sive Filter | Country Mosaic and OSM Labels |  Sentinel-2 Image List |
| -- | -- | -- | -- | -- |
| Austria(AU) | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | Total: **9 tiles** 085,095,107,108,122,124,138,139 |
| Belgium(BE) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **6 tiles** 065,066,072,073,078,079 |
| Bulgaria(BU) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **8 tiles** 183,184,201,202,219,220,234,235 |
| Croatia(CR) | :heavy_check_mark:| :heavy_check_mark: | :heavy_check_mark: | Total: **8 tiles** 107,121,122,136,137,150,151,152 |
| Cyprus(CY) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **2 tiles** 251,253  |
| Czechia(CZ) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **9 tiles** 109,110,123,124,125,138,140,154  |
| Denmark(DN) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **8 tiles** 089,090,099,100,111,112,113,127 |
| Estonia(ETO) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **8 tiles** 191,192,209,210,225,226,239,240 |
| Finland(FI) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **32 tiles** 174,175,176,179,192,193,194,196,197,210,211,212,213,214,215,226,227,228,229,230,231,240,241,242,243,244,245,247,248,249,250,252  |
| France(FR) | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark: |Total: **32 tiles** 020,021,031,032,041,042,043,044,053,054,055,056,057,061,062,063,064,065,066,069,070,071,072,075,076,077,078,082,083,085,086,092 |
| Germany(DE) | :heavy_check_mark:  | :heavy_check_mark:  | :heavy_check_mark:  |Total: **23 tiles**  077,078,079,080,085,086,087,088,089,095,096,097,098,099,108,109,110,111,112,123,124,125,126 |
| Greece(GR) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **16 tiles** 164,165,166,180,181,182,183,198,199,200,201,216,217,218,219,232  |
| Hungary(HU) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **7 tiles** 137, 138, 152, 153, 167, 168, 186 |
| Ireland(IR) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **8 tiles** 004,005,006,011,012,013,022,023  |
| Italy(IT) | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | Total: **28 tiles** 075,076,081,082,083,084,091,092,093,094,095,102,103,104,105,106,107,108,118,119,120,121,122,133,134,135,149,150 |
| Latvia(LA) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **10 tile** 172,173,190,191,208,209,224,225,238,239 |
| Lithuania(LI) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **7 tile** 172,189,190,207,208,223,224  |
| Malta(MA) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **1 tile** 117 |
| Netherlands(NL) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **5 tiles** 066,073,074,079,080 |
| Poland(PO) | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | Total: **19 tiles** 125,126,139,140,141,142,154,155,156,157,169,170,171,187,188,189,205,206,207 |
| Portugal(POR) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **7 tiles** 000,001,002,007,008,009,018 |
| Romania(RO) | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | Total: **15 tiles** 167,168,184,185,186,202,203,204,220,221,222,235,236,237,246 |
| Slovakia(SLK) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **8 tiles** 138,139,153,154,168,169,186,187|
| Slovenia(SLN) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **5 tiles** 107,122,123,137,138 |
| Spain(SP) | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: | Total: **26 tiles** 002,003,007,008,009,010,015,016,017,018,019,027,028,029,030,038,039,040,041,051,052,053,054,060,061,068 |
| Sweden(SW) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **36 tiles** 100,101,112,113,114,115,116,127,128,129,130,131,132,142,143,144,145,146,147,148,158,159,160,161,162,163,176,177,178,179,194,195,196,197,213,214 |
| United Kingdom(UK) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **25 tiles** 012,013,014,021,022,023,024,025,032,033,034,035,036,037,044,045,046,047,048,049,050,058,059,066,067 |
| Luxembourg(LU) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | Total: **1 tiles** 078 |

**All results availble in SDS under the OSM4EO folder**

- classified map file example: *OSM4EO\sepal.io\eu_countries_010_10m\S2_10m_3035_classified.tif*
- sivev classified map file example: *OSM4EO\sepal.io\eu_countries_010_10m\S2_10m_3035_classified_sieved.tif*
- country mosaic file example: *OSM4EO\mosaic_EU\DE_10m_3035_classified_sieved.tif*
- OSM label file example:*OSM4EO\label_EU\DE_10m_3035_label.tif*

### Preprocessing
 
The first step is to prepare the training samples from SDS label files. Within the code, one should define the training samples images list and output files.

```bash
$ python 01_preproce_prepare_sample.py
```

### Training

After preparing the training samples, one could start the model training. The current model is based on deep residual convolutional neural networks (ResNet), within this code, one could have the model depth and the locations of input and output files.

```bash
$ python 02_keras_models_single.py
```

### Prediction 

Once the model trianing is finished, this code will produce the classification results based on the trained model and images list. Within the code, one could change this trained model path and image index.

```bash
$ python 03_keras_classification.py
```

### Postprocessing

After the prediction, one could extract individual layer by implmenting this postprocessing code. Here as a example, we extract the class 12 (water) from the entire Germany.

```bash
$ python 04_postproce_extract_LU.py
```

### Others

To submit a bash job in bwcluster, one could submit this code.

```bash
$ msub bwcluster_bash_job.sh
```



## Related work:

Li, H.; Ghamisi, P.; Rasti, B.; Wu, Z.; Shapiro, A.; Schultz, M.; Zipf, A. A Multi-Sensor Fusion Framework Based on Coupled Residual Convolutional Neural Networks. Remote Sens. 2020, 12, 2067. https://doi.org/10.3390/rs12122067

