"""
This is a testing script for the found bug in opencv where using the set function on a video capture object may not set the frame correctly for unknown reasons. Instead reading the video from the beginning will result in the correct frame being found.

Two subdirectoreis of frame_dir will be created iterate, and seek. Each frame
extracted by iterating through each entire video will be stored in iterate, and each frame extracted by seeking to the same frame number will be stored in seek.

Found the bug to not be replicable using this script, it was determined that
different video files were given and that it was not an issue with the code but
the source files given.
"""
import extract_allframes as eaf

import os
import cv2
import glob

def seek_frame(vcap, frame_num):
    vcap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, im = vcap.read()
    return im


def main(args):
    iter_out = os.path.join(args.frame_dir, 'iterate')
    os.makedirs(iter_out, exist_ok=True)
    seek1_out = os.path.join(args.frame_dir, 'seek_in_order')
    os.makedirs(seek1_out, exist_ok=True)
    seek2_out = os.path.join(args.frame_dir, 'seek_jump')
    os.makedirs(seek2_out, exist_ok=True)
    seek3_out = os.path.join(args.frame_dir, 'seek_jump_back')
    os.makedirs(seek3_out, exist_ok=True)

    eaf.extract_all(args.video_dir, iter_out)

    for v in glob.glob(os.path.join(args.video_dir, '*')):
        bn, ext = os.path.splitext(os.path.basename(v))
        vcap = cv2.VideoCapture(v)
        i = 0
        ret = True
        vcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, im = vcap.read()
        while ret:
            out_name = f"{bn}-{i}.png"
            write_ret = cv2.imwrite(os.path.join(seek1_out, out_name), im)
            if not write_ret:
                print(f"failed to write {out_name}")
            i = i + 1
            vcap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, im = vcap.read()

    for v in glob.glob(os.path.join(args.video_dir, '*')):
        bn, ext = os.path.splitext(os.path.basename(v))
        vcap = cv2.VideoCapture(v)
        i = 0
        ret = True
        vcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, im = vcap.read()
        while ret:
            out_name = f"{bn}-{i}.png"
            write_ret = cv2.imwrite(os.path.join(seek2_out, out_name), im)
            if not write_ret:
                print(f"failed to write {out_name}")
            i = i + 1
            vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            vcap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, im = vcap.read()

    for v in glob.glob(os.path.join(args.video_dir, '*')):
        bn, ext = os.path.splitext(os.path.basename(v))
        vcap = cv2.VideoCapture(v)
        i = 0
        ret = True
        vcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, im = vcap.read()
        while ret:
            out_name = f"{bn}-{i}.png"
            write_ret = cv2.imwrite(os.path.join(seek3_out, out_name), im)
            if not write_ret:
                print(f"failed to write {out_name}")
            i = i + 1
            vcap.set(cv2.CAP_PROP_POS_FRAMES, 125)
            vcap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, im = vcap.read()

    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True,
                        help="""
                        Path to directory containing all video files that all frames should be extracted from.
                        """)
    parser.add_argument('--frame_dir', type=str, required=True, 
                        help="""
                        Directory where two sub directories will be created, iterate and seek. All extracted frames will be written to these two directories.
                        """)
    args = parser.parse_args()

    os.makedirs(args.frame_dir, exist_ok=True)
    main(args)