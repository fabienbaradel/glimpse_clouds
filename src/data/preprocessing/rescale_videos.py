import argparse
import os
import subprocess
import time
import sys
import ipdb
import pickle
from utils.meter import *


def main(args):
    # Parameters from the args
    dir, h, w, fps, common_suffix = args.dir, args.height, args.width, args.fps, args.common_suffix

    # avi dir
    dir_split = dir.split('/')
    avi_dir = dir_split[-1]
    root_dir = '/'.join(dir_split[:-1])
    new_avi_dir = "{}_{}x{}_{}".format(avi_dir, w, h, fps)
    new_dir = os.path.join(root_dir, new_avi_dir)
    os.makedirs(new_dir, exist_ok=True)

    # load the existing dict if exist
    dict_video_length_fn = os.path.join(new_dir, 'dict_id_length.pickle')
    if os.path.isfile(dict_video_length_fn):
        with open(dict_video_length_fn, 'rb') as file:
            dict_video_length = pickle.load(file)
    else:
        dict_video_length = {}

    # Get the super_video filenames
    list_video_fn = get_all_videos(dir, common_suffix)

    print("{} videos to uncompressed in total".format(len(list_video_fn)))

    # Loop over the super_video and extract
    op_time = AverageMeter()
    start = time.time()
    list_error_fn = []
    for i, video_fn in enumerate(list_video_fn):
        try:
            # Rescale
            rescale_video(video_fn, w, h, fps, dir, new_dir, common_suffix, dict_video_length, ffmpeg=args.ffmpeg,
                          crf=args.crf)

            # Log
            duration = time.time() - start
            op_time.update(duration, 1)
            print("{}/{} : {time.val:.3f} ({time.avg:.3f}) sec/super_video".format(i + 1, len(list_video_fn),
                                                                                   time=op_time))
            sys.stdout.flush()
            start = time.time()
        except:
            print("Impossible to rescale_videos super_video for {}".format(video_fn))
            list_error_fn.append(video_fn)

    print("\nDone")
    print("\nImpossible to extract frames for {} videos: \n {}".format(len(list_error_fn), list_error_fn))

    # Save the dict id -> length
    with open(dict_video_length_fn, 'wb') as file:
        pickle.dump(dict_video_length, file, protocol=pickle.HIGHEST_PROTOCOL)
        print("\nDict Video_id -> Length saved here ---> {}".format(file))


def get_duration(file):
    """Get the duration of a super_video using ffprobe. -> https://stackoverflow.com/questions/31024968/using-ffmpeg-to-obtain-super_video-durations-in-python"""
    cmd = 'ffprobe -i {} -show_entries format=duration -v quiet -of csv="p=0"'.format(file)
    output = subprocess.check_output(
        cmd,
        shell=True,  # Let this run in the shell
        stderr=subprocess.STDOUT
    )
    # return round(float(output))  # ugly, but rounds your seconds up or down
    return float(output)


def rescale_video(video_fn, w, h, fps, dir, new_dir, common_suffix, dict_video_length, ffmpeg, crf=17):
    """ Rescale a super_video according to its new width, height an fps """

    # Output video_name
    video_id = video_fn.replace(dir, '').replace(common_suffix, '')
    video_fn_rescaled = video_fn.replace(dir, new_dir)
    video_fn_rescaled = video_fn_rescaled.replace(common_suffix, common_suffix.lower())

    # Create the dir
    video_dir_to_create = '/'.join(video_fn_rescaled.split('/')[:-1])
    os.makedirs(video_dir_to_create, exist_ok=True)

    # Check if the file already exists
    if os.path.isfile(video_fn_rescaled):
        print("{} already exists".format(video_fn_rescaled))
    else:
        subprocess.call(
            '{ffmpeg} -i {video_input} -vf scale={w}:{h} -crf {crf} -r {fps} -y {video_output} -loglevel panic'.format(
                ffmpeg=ffmpeg,
                video_input=video_fn,
                h=h,
                w=w,
                fps=fps,
                video_output=video_fn_rescaled,
                crf=crf
            ), shell=True)

        # Get the duration of the new super_video (in sec)
        duration_sec = get_duration(video_fn_rescaled)
        duration_frames = int(duration_sec * fps)

        # update the dict id -> length
        dict_video_length[video_id] = duration_frames

    return video_fn_rescaled


def get_all_videos(dir, extension='mp4'):
    """ Return a list of the super_video filename from a directory and its subdirectories """

    list_video_fn = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in [f for f in filenames if f.endswith(extension)]:
            fn = os.path.join(dirpath, filename)
            list_video_fn.append(fn)

    return list_video_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess dataset')
    parser.add_argument('--dir', metavar='DIR',
                        default='/Users/fabien/Datasets/NTU-RGB-D/avi',
                        help='path to avi dir')
    parser.add_argument('--width', default=256, type=int,
                        metavar='W', help='Width')
    parser.add_argument('--height', default=256, type=int,
                        metavar='H', help='Height')
    parser.add_argument('--fps', default=30, type=int,
                        metavar='FPS',
                        help='Frames per second for the extraction, -1 means that we take the fps from the super_video')
    parser.add_argument('--common-suffix', metavar='E',
                        default='_rgb.avi',
                        help='Common end of each super_video file')
    parser.add_argument('--crf', default=17, type=int,
                        metavar='CRF',
                        help='CRF for ffmpeg command')
    parser.add_argument('--ffmpeg', metavar='FF',
                        default='ffmpeg',
                        help='ffmpeg verison to use')

    args = parser.parse_args()

    main(args)
