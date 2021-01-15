# Badminton shuttle tracking
Attempts to track the position of the shuttle(s) in a video.

## Usage summary


```
# Dataset specific commands
# Run once to load data into appropriate locations

# Clean images and annotation files
python clean.py BADMINTON_DATA_ROOT

# Copy data to the preset location
python prepare.py --src BADMINTON_DATA_ROOT
# Alternatively, just copy a subset
python prepare.py --src BADMINTON_DATA_ROOT --include 01,02,05



# Training

# Test train split
python split.py

# Actual training and evaluation
python main.py
# Or start from an intermediate step with different arguments
python main.py --start_from eval --iou_thres 0.25
```

## Data preparation
### File layout
The paths for the input are specified by `--frames_path` and `--xml_path`. The files should have the following structure:
```
ROOT/
    VID1/
        frame_001
        frame_002
        ...
    VID2/
        frame_001
        frame_002
        ...
    ...
```
The filename doesn't matter as long as they are ordered correctly.

### Test train split
For training, the data are split into training and testing sets. This is done by specifying the filenames in files `--training_set` and `--testing_set`.

The files can be created manually or generated with `split.py`.

### Exclusion area
To avoid mis-labeling shuttles rolling on the floor as "non-ball", the program accepts coordinates of the area where shuttles might land on. The coordinates are specified in JSON format, e.g.
```js
{
    "VID1": [x1, y1, x2, y2],
    "VID2": [x1, y1, x2, y2],
    ...
}
```

The default location of the JSON file is `data/exclude.json`
