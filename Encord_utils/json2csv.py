"""
Script to read a given labels.json file, and output two csv files that contain
the relevant information for the created labels.

#TODO: Generalize to encord exported json files that aren't retraction.
#TODO: Generalize to encord exported json files that aren't classification.
"""
import json
from collections.abc import Sequence, Mapping
from collections import defaultdict
import pandas as pd
from glob import glob
import os
import os.path as osp
import datasets
import draw_segmentation as ds
import cv2
import numpy as np
import colour_maps as colours
from tqdm import tqdm


def add_classes(seg_df, obj_classes):
    """
    Helper function that iterates over all classes for the latest segmentation to be added to seg_df, and does three things. 
        1. creates a new column for any class that isn't represented yet,
            backfilling with empty strings for all previous entries.
        2. appends the assigned values of all classes
        3. appends an empty string to any classes that exist but doesn't have an
            assignment.
    All changes are done to seg_df directly, i.e. inplace.
    args:
        seg_df: dict, that will be transformed into a pandas dataframe, so
            all values must stay at the same length. Assumes all entries have
            already been appended except any classes, so the length of any entry
            should only differ by 1. Assumes title is never the name of a class.
        obj_classes: dict, with keys being the classification name, and values
            being the classification value(s)
    Returns None
    """
    for oClass in obj_classes:
        if oClass not in seg_df:
            seg_df[oClass] = ['']*(len(seg_df['title']) - 1)
            seg_df[oClass].append(obj_classes[oClass])
    
    for k in seg_df:
        if len(seg_df[k]) == len(seg_df['title']) - 1:
            seg_df[k].append('')
        elif len(seg_df[k]) != len(seg_df['title']):
            msg = f"Found discrepancy in the number of entries for seg_df. Entry {k} was not equal in length to the number of entries in seg_df['title'], and was not short by a single entry."
            raise ValueError(msg)
    
    return None
    
            
def unpack(col):
    assert len(col) == 1
    if isinstance(col, Sequence):
        return col[0]
    if isinstance(col, Mapping):
        return list(col.items())[0]
    raise ValueError(f"Unable to unpack the given col: {col}")


def main(args):
    with open(args.json_file, 'r') as f:
        all_data = json.load(f)

    dataset_obj = datasets.dataset_switch(args.dataset)
    data_names = dataset_obj.get_names()

    normalized_names = {dataset_obj.norm_fn(n): n for n in data_names}

    anomalies = defaultdict(list)
    processed_classes = []
    processed_objects = []
    for video_data in all_data:
        cont = False
        title = dataset_obj.norm_fn(video_data["data_title"])
        try:
            our_v_name = normalized_names[title]
        except:
            anomalies["our_not_found"].append(title)
            continue
        data_unit = unpack(video_data["data_units"])[1]
        classification_answers = video_data["classification_answers"]
        object_answers = video_data["object_answers"]
        if (data_fps := data_unit["data_fps"]) > 1000:
            anomalies["high_fps"].append((title, data_fps))
            continue
        labels = data_unit["labels"]
        if len(labels) < 1:
            anomalies["short_labels"].append((title, len(labels), data_fps))
            continue
        data_duration = data_unit["data_duration"]
        ## debug section
        #for k, v in labels.items():
        #    v_c = v['classifications']
        #    for c in v_c:
        #        class_name = c['name']
        #        classification_tmp = classification_answers[c['classificationHash']]['classifications']
        #        classifications_len = len(classification_tmp)
        #        if classifications_len != 1:
        #            print(f"{class_name}: class_len-{classifications_len}")
        #            continue
        #        answers_len = len(unpack(classification_tmp)['answers'])
        #        if answers_len != 1:
        #            import pdb
        #            pdb.set_trace()
        #            print(f"{class_name}: class_len-{classifications_len}, ans_len-{answers_len}")
        #            continue
        #        name_tmp = unpack(unpack(classification_tmp)['answers'])['name']
        #        print(f"{class_name}: cls_name-{name_tmp}, class_len-{classifications_len}, ans_len-{answers_len}")
        ##
        classifications = defaultdict(dict)
        objects = defaultdict(list)
        objHashes = defaultdict(list)
        obj_classifications = defaultdict(list)
        for k, v in labels.items():
            v_c = v['classifications']
            v_o = v['objects']
            for c in v_c:
                class_tmp = unpack(
                    classification_answers[c['classificationHash']]['classifications']
                    )
                if isinstance(class_tmp['answers'], list):
                    ans_tmp = unpack(class_tmp['answers'])['name']
                elif isinstance(class_tmp['answers'], str):
                    ans_tmp = class_tmp['answers']
                else:
                    msg = (f"{c['name']}: Unimplemented type for answers"
                           f" values, {class_tmp['answers']}")
                    raise NotImplementedError(msg)
                classifications[int(k)][c["name"]] = ans_tmp
                #classifications[int(k)][c["name"]] = unpack(
                #    unpack(
                #        classification_answers[c["classificationHash"]]["classifications"]
                #    )["answers"]
                #)["name"]
            for o in v_o:
                objects[int(k)].append(ds.EncordObject(o))
                objHashes[int(k)].append(o['objectHash'])
                try:
                    this_obj_cls = object_answers[o['objectHash']]
                except KeyError as e:
                    this_obj_ans = {}
                if this_obj_cls:   
                    # Pairing down the extraneous information
                    this_obj_cls = this_obj_cls['classifications']
                    this_obj_ans = {}
                    for oc in this_obj_cls:
                        ans_names = [oca['name'] for oca in oc['answers']]
                        # Handle case when there is only a single answer,
                        # otherwise it should stay a list.
                        if len(ans_names) == 1:
                            ans_names = ans_names[0]
                        this_obj_ans[oc['name']] = ans_names
                obj_classifications[int(k)].append(this_obj_ans)
                if args.colour_map:
                    setattr(objects[int(k)][0], 'color', 
                            getattr(colours, args.colour_map)[o['color']])
            if len(set(objHashes[int(k)])) != len(objects[int(k)]):
                msg = (f"Assumption that each segmentation in a frame has a "
                       f"unique hash has been broken for {title}-{k}")
                raise ValueError(msg)

        if cont:
            anomalies["incomplete_answers"].append((title, v_c))
            cont = False
            continue
        processed_classes.append((title, our_v_name, data_duration, data_fps, classifications))
        processed_objects.append((title, our_v_name, objects, objHashes, obj_classifications))

    ### Build a hash mapping that will handle windows File Systems illegal
    #characters, creates a mapping to legal characters not already seen in the
    #current set of hashes, if there are no unused characters double characters
    #will be added.
    # This is a superset of linux chars
    windows_illegal_chars = set('/:?<>"|\\*')
    all_hashes_lst = []
    all_hash_chars = []
    for po in processed_objects:
        for k,v in po[3].items():
            all_hashes_lst = all_hashes_lst + v
            for h in v:
                for c in h:
                    all_hash_chars.append(c)
    all_hashes = set(all_hashes_lst)
    if len(set([len(h) for h in all_hashes])) != 1:
        msg = f"objectHashes are of multiple lengths."
        raise NotImplementedError(msg)
    all_hash_chars = set(all_hash_chars)
    need_replace = all_hash_chars.intersection(windows_illegal_chars)
    pos_repl =('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
               '_- ()+=!@#$%^&')
    hash_map = {}
    i = 1
    while need_replace:
        for nr in need_replace:
            for pr in pos_repl:
                if not pr*i in all_hash_chars:
                    hash_map[nr] = pr*i
                    all_hash_chars = all_hash_chars.union(set([pr*i]))
                    need_replace = need_replace.difference(set(nr))
                    all_hash_chars = all_hash_chars.difference(set(nr))
                    break
        i = i + 1
    trans_map = str.maketrans(hash_map)

    # Write anomalies to files
    #TODO: Change this to not write files if the dataframes are empty?
    not_found_path = osp.join(args.out_dir, 'anomaly_not_found.csv')
    not_found_cols = ['video']
    pd.DataFrame(anomalies['our_not_found'], 
                 columns=not_found_cols).to_csv(not_found_path, index=False)
    high_fps_path = osp.join(args.out_dir, 'anomaly_high_fps.csv')
    high_fps_cols = ['video', 'fps']
    pd.DataFrame(anomalies['high_fps'],
                 columns=high_fps_cols).to_csv(high_fps_path, index=False)
    short_lbls_path = osp.join(args.out_dir, 'anomaly_short_labels.csv')
    short_lbls_cols = ['video', 'n labels', 'fps']
    pd.DataFrame(anomalies['short_labels'],
                 columns=short_lbls_cols).to_csv(short_lbls_path, index=False)
    incomplete_path = osp.join(args.out_dir, 'anomaly_incomplete.csv')
    incomplete_cols = ['video', 'data']
    pd.DataFrame(anomalies['incomplete_answers'],
                 columns=incomplete_cols).to_csv(incomplete_path, index=False)

    questions = set()
    for *_, classifications in processed_classes:
        questions.update((k_ for k, v in classifications.items() for k_ in v.keys()))
    questions = sorted(list(questions))

    videos_df = []
    frames_df = []
    # Steps through all frames for each video info extracted from the json
    for title, our_v_name, duration, fps, classifications in processed_classes:
        videos_df.append((title, our_v_name, duration, fps))
        for k, v in classifications.items():
            all_lbls = []
            for q in questions:
                try:
                    all_lbls.append(v[q])
                except KeyError:
                    all_lbls.append(None)
            frames_df.append((title, k, *(all_lbls)))
    videos_df = pd.DataFrame(
        videos_df, columns=("title", "our_name", "duration", "fps")
    )
    frames_df = pd.DataFrame(
        frames_df,
        columns=["title", "frame_number"] + questions
    )

    frames_df = dataset_obj.norm_df(frames_df)

    videos_df.to_csv(osp.join(args.out_dir, "videos.csv"), index=False)
    frames_df.to_csv(osp.join(args.out_dir, "frames.csv"), index=False)

    ### output all segmentations
    segmentation_out = os.path.join(args.out_dir, 'segmentations')
    os.makedirs(segmentation_out, exist_ok=True)
    # raw frames with no segmentations
    raw_dir = os.path.join(segmentation_out, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    # visual of the segmentations
    over_dir = os.path.join(segmentation_out, 'overlaid')
    os.makedirs(over_dir, exist_ok=True)
    # isolated segmentations
    iso_dir = os.path.join(segmentation_out, 'isolated')
    os.makedirs(iso_dir, exist_ok=True)

    # seg_df is a summary dataframe that will contain all the information about
    # each segmentation put on a frame, and any classifications those
    # segmentations have been assigned.
    seg_df = {'title': [], 'frame_num': [], 'segmentation name':[], 'color': [],
              'shape': [], 'points': [], 'objectHash': [], 'fileHash': []}
    # all_obj_classes, keep a running tally of the new columns added to seg_df
    # which will be the name of the new class and the value assigned to it for
    # each frame.
    failed_read_path = osp.join(args.out_dir, 'anomaly_failed_read.csv')
    failed_read_dict = {'video_name': [], 'frame_num': [], 'path': []}
    for po in tqdm(processed_objects):
        title = po[0]
        our_v_name = po[1]
        vid_objects = po[2]
        obj_hashes = po[3]
        obj_classes = po[4]

        vid_path = dataset_obj.get_video_path(our_v_name)

        cap = cv2.VideoCapture(vid_path)
        for frame_num, frame_objects in vid_objects.items():
            # Encord hash
            ohashes = obj_hashes[frame_num]
            # file name appropriate hash, also addes the frame number since
            # ohashes link the "same" segmentation through a video.
            fhashes = [ohash.translate(trans_map) + f"-{frame_num}" for ohash in ohashes]

            oclasses = obj_classes[frame_num]

            out_path = os.path.join(over_dir, f"{title}-{frame_num}.png")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, bg = cap.read()
            if not ret:
                #err_msg = f"Failed to read {frame_num=} from {our_v_name=}"
                #raise ValueError(err_msg)
                # debug
                failed_read_dict['video_name'].append(our_v_name)
                failed_read_dict['frame_num'].append(frame_num)
                failed_read_dict['path'].append(vid_path)
            else:
                try:
                    cv2.imwrite(os.path.join(raw_dir, f"{title}-{frame_num}.png"), bg)
                except:
                    import pdb
                    pdb.set_trace()
                for fo, oh, fh, oc in zip(frame_objects, ohashes, fhashes, oclasses):
                    seg_df['title'].append(title)
                    seg_df['frame_num'].append(frame_num)
                    seg_df['segmentation name'].append(fo.name)
                    seg_df['color'].append(fo.color)
                    seg_df['shape'].append(fo.shape_type)
                    seg_df['points'].append(fo.shape)
                    seg_df['objectHash'].append(oh)
                    seg_df['fileHash'].append(fh)
                    add_classes(seg_df, oc)
                    bg = fo.draw(bg, alpha=0.4)
                    iso_bg = np.zeros(bg.shape)
                    iso_bg = fo.draw(iso_bg, color=(255, 255, 255), alpha=1)
                    # ObjHashes are not unique across a video, so append the
                    # frame number to distinguish it from the same segmentation
                    # on different frames.
                    iso_out = os.path.join(iso_dir, fh + '.png')
                    ret = cv2.imwrite(iso_out, iso_bg)
                ret = cv2.imwrite(out_path, bg)
                if not ret:
                    err_msg = f"Failed to write {out_path}"
                    raise ValueError(err_msg)
        cap.release()
    pd.DataFrame(seg_df).to_csv(os.path.join(segmentation_out, 'details.csv'),
                                index=False)
    pd.DataFrame(failed_read_dict).to_csv(failed_read_path, index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, required=True,
                        help="""
                        Path to json file to be read in.
                        """)
    parser.add_argument('--out_dir', type=str, default='.',
                        help="""
                        Path to directory where output csv files should be
                        written. Default is the working directory.
                        """)
    dataset_choices = datasets.dataset_switch.valid_datasets
    parser.add_argument('--dataset', type=str, required=True,
                        choices=dataset_choices,
                        help="""
                        Dataset defined in datasets.py that defines where the
                        source data is stored on H4h, how to match the filenames
                        with the data_titles in the labels.json, and how to
                        normalize the labels before writing them to a csv file.
                        """)
    colour_map_choices = [c for c in dir(colours) if c[0] != '_']
    parser.add_argument('--colour_map', choices=colour_map_choices,
                        default=None,
                        help="""
                        A way to select from some harcoded colour mappings to
                        override the colour mappings in the json. Implemented
                        mappings are found in colour_maps.py. This will also
                        change the colour recorded in the details.csv file.
                        """)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    main(args)