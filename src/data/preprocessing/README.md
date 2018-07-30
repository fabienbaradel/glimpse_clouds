Rescaling of a video dataset
======================================================

Rescale the video of an entire directory

## The NTU dir --- Sanity check
Below we assume that you have already download the NTU dataset and store it in a directory (e.g `/Users/fabien/Datasets/NTU-RGB-D`).
The videos needs to be stored in in the subdirectories called `avi` (e.g. full path in my case `/Users/fabien/Datasets/NTU-RGB-D/avi`).
And the skeleton files need to be stored in the subdirectories `skeleton`.             
             
## Command line
Below is an example of command line for rescaling the entire NTU dataset (resolution from 1920x1080 to 256x256).

```shell
# Path to your ffmpeg
ffmpeg_version=ffmpeg

# Python command line
python rescale_videos.py \
--dir /Users/fabien/Datasets/NTU-RGB-D/avi \
--width 256 \
--height 256 \
--fps 30 \
--common-suffix _rgb.avi \
--crf 17 --ffmpeg ffmpeg
```
Running the above lines should create a directory ```/Users/fabien/Datasets/NTU-RGB-D/avi_256x256_30``` containing all the videos rescaled to 256X256 with a fps of 30.
You should also have in the directory a file named ```dict_id_length.pickle``` which contains the number of frames of each video.
It would be useful when extracting clip from the whole video while training.

This rescaling step allows to reduce the size of the avi files from 127G to 8.4G (1920x1080 -> 256x256)
```shell
# Size of the initial directory
du -sh /Users/fabien/Datasets/NTU-RGB-D/avi

# Size of the rescaled directory
du -sh /Users/fabien/Datasets/NTU-RGB-D/avi_256x256_30

# Open a video
open /Users/fabien/Datasets/NTU-RGB-D/avi_256x256_30/S017C003P008R002A001_rgb.avi
```

