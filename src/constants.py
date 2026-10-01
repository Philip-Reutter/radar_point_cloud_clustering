SCENE = 1                        # choose scene number for clustering comparison
HDF5_FILE = r"ENTER_PATH_HERE"   # eg .../RadarScenes/data/sequence_{SCENE}/radar_data.h5
JSON_FILE = r"ENTER_PATH_HERE"   # eg .../RadarScenes/data/sequence_{SCENE}/scenes.json
IMAGES_PATH = r"ENTER_PATH_HERE" # eg .../RadarScenes/data/sequence_{SCENE}/camera"

COLOR_CLASS_DICT = {
    0: ('car', 'blue'),
    1: ('large vehicle', 'green'),
    2: ('large vehicle', 'green'),
    3: ('large vehicle', 'green'),
    4: ('large vehicle', 'green'),
    5: ('two-wheeler', 'yellow'),
    6: ('two-wheeler', 'yellow'),
    7: ('pedestrians', 'pink'),
    8: ('pedestrians group', 'purple'),
    9: ('static/other', 'black'),
    10: ('static/other', 'black'),
    11: ('static/other', 'black'),
}
