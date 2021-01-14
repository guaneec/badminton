import xml.etree.ElementTree as ET


def read_boxes(annotation_file):
    root = ET.parse(annotation_file)
    boxes = []
    for b in root.findall("object/bndbox"):
        xmin = int(b.find("xmin").text)
        xmax = int(b.find("xmax").text)
        ymin = int(b.find("ymin").text)
        ymax = int(b.find("ymax").text)
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def fix_conv():
    # fix "No algorithm worked" error
    # https://github.com/tensorflow/tensorflow/issues/43174
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)