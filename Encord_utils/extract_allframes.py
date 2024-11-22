import cv2
import os
import glob
import pandas as pd

def extract_all(video_dir, frame_dir, video_glob='*'):
    """
    Extract all frames from the videos and write to png files.
    """
    for path in glob.glob(os.path.join(video_dir, video_glob)):
        v = cv2.VideoCapture(path)
        fn = os.path.basename(path)
        bn, ext = os.path.splitext(fn)
        frame_ext = '.png'
        i = 0
        ret, im = v.read()
        while ret:
            frame_path = os.path.join(frame_dir, bn + '-' + str(i) + frame_ext)
            w_ret = cv2.imwrite(frame_path, im)
            if not w_ret:
                print(f"problem writing frame {frame_path}")
            i = i + 1
            ret, im = v.read()
        print(f"Done video {fn}")
    return None


def extract_by_df(args):
    """
    Given a csv of frames that are labelled, extract only the frames listed in
    the csv file. Assumes the existant of two columns title and frame_number.
    """
    df = pd.read_csv(args.frames_csv)
    all_videos = os.listdir(args.video_dir)
    t2v_map = {os.path.splitext(v)[0].lower():v for v in all_videos}
    titles = [os.path.splitext(t)[0].lower() for t in df['title'].unique()]

    for t in titles:
        try:
            path = os.path.join(args.video_dir, t2v_map[t])
        except KeyError:
            print(f"Missing video for title {t}")
            continue
        frame_nums = df[df['title'] == t]['frame_number'].unique()
        frame_nums = sorted(frame_nums)
        frame_ext='.png'
        v = cv2.VideoCapture(path)
        for frame_n in frame_nums:
            fn = t + '-' + str(frame_n) + frame_ext
            frame_path = os.path.join(args.frame_dir, fn)
            v.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
            ret, im = v.read()
            w_ret = cv2.imwrite(frame_path, im)
            if not w_ret:
                print(f"problem writing frame {frame_path}")
        print(f"Done video {t}")
    return None


def main(args):

    if not args.frames_csv:
        extract_all(args.video_dir, args.frame_dir, args.video_glob)
    else:
        extract_by_df(args)

    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True,
                        help="""
                        Video files that all frames should be extracted from.
                        """)
    parser.add_argument('--frame_dir', type=str, required=True, 
                        help="""
                        Directory where all extracted frames will be written.
                        """)
    parser.add_argument('--video_glob', type=str, required=True,
                        help="""
                        Glob to match videos in video_dir.
                        """)
    parser.add_argument('--frames_csv', type=str, default=None,
                        help="""
                        Path to csv file with the frames that should be extracted from the video.
                        """)
    args = parser.parse_args()

    os.makedirs(args.frame_dir, exist_ok=True)
    main(args)