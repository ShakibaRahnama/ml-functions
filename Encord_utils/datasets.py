"""
Container for helper functions that load specific datasets.
"""
import os.path as osp
import os
import glob
import re
from abc import ABC, abstractmethod

def dataset_switch(dataset):
    """
    Simple switch to return the functions needed for json2csv for a given dataset.

    Args:
        dataset: str, name of the dataset that the utility functions should be
            returned.
    Returns:
        implementation of the abc dataset as found in this script.
    """
    match dataset:
        case 'ALL_FRAMES_subvideos':
            data_obj = ALL_FRAMES_subvideos()
        case 'Colorectal_ret':
            data_obj = Colorectal_ret()
        case 'Adrenal_left':
            data_obj = Adrenal_left()
        case 'Pituitary':
            data_obj = Pituitary()
        case 'Colorectal_Go':
            data_obj = Colorectal_Go()
        case 'Colorectal_Go_raw':
            data_obj = Colorectal_Go_raw()
        case 'Bariatric':
            data_obj = Bariatric()
        case 'tool_measurement':
            data_obj = tool_measurement()
        case 'ziyad_test':
            data_obj = ziyad_test()
        case 'ziyad_tti':
            data_obj = ziyad_tti()
    
    return data_obj

# this attribute of dataset_switch might be extraneous, is there a clever way to
# grab all subclasses of dataset instead?
dataset_switch.valid_datasets = ["ALL_FRAMES_subvideos", "Colorectal_ret",
                                 "Adrenal_left", "Pituitary", "Colorectal_Go",
                                 "Colorectal_Go_raw", "Bariatric",
                                 "tool_measurement", "ziyad_test", "ziyad_tti",
                                 "ziyad_tti_check"]


class dataset(ABC):
    """
    TODO: implement the __init__ method, which will force the inheriting classes
    to have all the required attributes.
    """
    @abstractmethod
    def norm_fn(self, fn):
        """
        Matches the file names to the data_titles in the labels.json file
        provided from Encord.
        """
        ...
    
    @abstractmethod
    def get_names(self):
        """
        Get all filenames for the videos or base files that the labels were applied to. Should have copies of that original data on H4H.
        """
        ...
    
    @abstractmethod
    def norm_df(self, df):
        """
        Normalize all columns for the given pandas DataFrame. Normalize usually means to change the strings to integers.
        """
        ...
    

class ALL_FRAMES_subvideos(dataset):
    def __init__(self):
        self.name = 'ALL_FRAMES_subvideos'
        self.glob = '/cluster/projects/madanigroup/CLAIM/subvideos/downscaled/*'
        self.classification_types = ['retraction', 'exposure_ieg', 'exposure_los']

    def norm_fn(self, fn):
        """
        Get a function that normalizes the dataset filenames.

        Returns:
            function(str) -> str, function that normalizes a filename to the same
                standard as found in the json file.
        """
        return fn.lower().replace("_", " ").replace("-", " ")

    def get_names(self):
        """
        Get all filenames for dataset ALL_FRAMES_subvideos.

        Returns:
            list of str, all filenames used in the dataset.
        """
        return [osp.splitext(osp.basename(p))[0] for p in glob.glob(self.glob)]
    
    def norm_df(self, df):
        """
        Normalize the label columns extracted from the labels.json file for the subvideo dataset. See self.label_types for all labels normalized.
        """
        df["retraction"] = df["retraction"].apply(lambda x: int(x[0]))
        df["exposure_ieg"] = df["exposure_ieg"].apply(
            lambda x: {"Yes": 1, "No": 0}[x]
        )
        df["exposure_los"] = df["exposure_los"].apply(
            lambda x: {"Yes": 1, "No": 0}[x]
        )
        return df


class Colorectal_ret(dataset):
    def __init__(self):
        self.name = 'Colorectal'
        self.glob = ('/cluster/projects/madanigroup/colorectal tme/colorectal tme/*')
        self.classification_types = ['retraction']
    
    def norm_fn(self, fn):
        bn, ext = osp.splitext(fn)
        return bn.replace('_normalized', '')
    
    def get_names(self):
        return [osp.basename(p) for p in glob.glob(self.glob)]
    
    def norm_df(self, df):
        df['retraction'] = df['retraction'].apply(lambda x: int(x[0]))
        return df


class Colorectal_Go(dataset):
    def __init__(self):
        self.name = 'Colorectal'
        self.glob = ('/cluster/projects/madanigroup/colorectal tme/colorectal tme/*')
        self.raw_regex = ('/cluster/projects/madanigroup/colorectal tme/'
                         'colorectal tme/.*mp4')
        self.regex = re.compile(self.raw_regex)
        self.is_vid_map_init = False
        self.vid_map = {}
    
    def norm_fn(self, fn):
        bn, ext = osp.splitext(fn)
        return bn.replace('_normalized', '')
    
    def get_names(self):
        return [osp.basename(p) for p in glob.glob(self.glob)]
    
    def norm_df(self, df):
        return df
    
    def get_video_path(self, video_name):
        video_path = os.path.join(self.glob[:-1], video_name)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Failed to find file {video_path}")
        return video_path


class Colorectal_Go_raw(dataset):
    def __init__(self):
        self.name = 'Colorectal'
        self.glob = ('/cluster/projects/madanigroup/CLAIM/Colorectal/videos_were_labelled/*')
        self.raw_regex = ('/cluster/projects/madanigroup/CLAIM/Colorectal'
                          '/raw_videos/.*mp4')
#        self.raw_regex = ('/cluster/projects/madanigroup/CLAIM/Colorectal'
#                          '/videos_were_labelled/.*mp4')
        self.regex = re.compile(self.raw_regex)
        self.is_vid_map_init = False
        self.vid_map = {}
        self.found_normalized_out = '/cluster/projects/madanigroup/CLAIM/Colorectal/normalized_Go_zones.txt'
        #Clear normalized_out file.
        f = open(self.found_normalized_out, 'w')
        f.close()
    
    def norm_fn(self, fn):
        bn, ext = osp.splitext(fn)
        if '_normalized' in bn:
            with open(self.found_normalized_out, 'a') as f:
                f.write(fn + '\n')
        return bn.replace('_normalized', '')
    
    def get_names(self):
        return [osp.basename(p) for p in glob.glob(self.glob)]
    
    def norm_df(self, df):
        return df
    
    def get_video_path(self, video_name):
        video_path = os.path.join(self.glob[:-1], video_name)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Failed to find file {video_path}")
        return video_path


class Adrenal_left(dataset):
    def __init__(self):
        self.name = 'Adrenal Left'
        self.glob = ('/cluster/projects/madanigroup/adrenal_left/data/videos/*')
        self.classification_types = []

    def norm_fn(self, fn):
        bn, ext = osp.splitext(fn)
        return bn

    def get_names(self):
        return [osp.basename(p) for p in glob.glob(self.glob)]
    
    def norm_df(self, df):
        df['Retraction Quality'] = df['Retraction Quality'].apply(lambda x: int(x[0]) if not x is None else x)
        phase_name_map = {'Phase 1 - Initial Access': 'Phase 1',
                       'Phase 2 - Mobilization of Kidney and Inf. Med. dissection of Adr / PeriAdr Fat ': 'Phase 2',
                       'Phase 3 - Adr V. dissection (RIGHT ONLY) ': 'Phase 3',
                       'Phase 4': 'Phase 4'}
        df = df.rename(columns=phase_name_map)
        for p in phase_name_map.values():
            df[p] = ~df[p].isnull()
        
        return df


class Pituitary(dataset):
    def __init__(self):
        self.name = 'Pituitary'
        self.glob = ('/cluster/projects/madanigroup/pituitary*processed/*')
        self.raw_regex = '/cluster/projects/madanigroup/pituitary.*processed/.*(MP4|mpg)'
        self.regex = re.compile(self.raw_regex)
        self.classification_types = []
        self.is_vid_map_init = False
        self.vid_map = {}

    def norm_fn(self, fn):
        fn = fn.replace('_normalized', '').lower()
        bn, ext = osp.splitext(fn)
        return bn
    
    def get_names(self):
        return [osp.basename(p) for p in glob.glob(self.glob) 
                if self.regex.match(p)]
    
    def get_video_path(self, video_name):
        if not self.is_vid_map_init:
            self.vid_paths = [p for p in glob.glob(self.glob) if self.regex.match(p)]
            for vid_path in self.vid_paths:
                fn = osp.basename(vid_path)
                self.vid_map[fn] = vid_path
            self.is_vid_map_init = True
        return self.vid_map[video_name]

    def norm_df(self, df):
        return df


class Bariatric(dataset):
    """
    Dataset object for Bariatric surgical videos dataset.
    """
    def __init__(self):
        self.name = 'Bariatric'
        self.base_dir = '/cluster/projects/madanigroup'
        #Notes:
        #1. directory 240314datY7grV-processed is a strict subset of
        # 240314datY7grV-231220vEzPmNTt-processed so the former is excluded.
        #2. 231201bmzyrmrN-processed contains many subdirectories, currently all
        # regular videos are under two directories and have the mp4 extension.
        # Uses this assumption, no checks made that this holds currently.
        self.globs = (os.path.join(self.base_dir,
                                   '240314datY7grV-231220vEzPmNTt-processed/*'),
                      os.path.join(self.base_dir,
                                   '231201bmzyrmrN-processed/**/**/*.mp4')
                      )
        self.is_vid_map_init = False

    def norm_fn(self, fn):
        bn, _ = os.path.splitext(fn)
        return bn
    
    def get_names(self):
        paths = []
        for g in self.globs:
            paths = paths + glob.glob(g)
        return [osp.basename(p) for p in paths]
    
    def get_video_path(self, video_name):
        if not self.is_vid_map_init:
            self.vid_paths = []
            for g in self.globs:
                self.vid_paths = self.vid_paths + glob.glob(g)
            self.vid_map = {osp.basename(vp):vp for vp in self.vid_paths}
            self.is_vid_map_init = True
        return self.vid_map[video_name]

    def norm_df(self, df):
        return df


class tool_measurement(dataset):
    """
    Dataset for Raphael's project to segmenting the edges of surgical
    instruments to be used in down stream calculations for returning the
    measurement between the two tool tips.
    """
    def __init__(self):
        self.name = 'tool measurement'
        self.base_dir = '/cluster/projects/madanigroup'
        self.globs = (os.path.join(self.base_dir,
                                   '231220vEzPmNTt-processed/*.mp4'),
                      os.path.join(self.base_dir,
                                   '240110tIVIw5gx-TME_opensource_01162024'
                                   '-processed/*.mp4'),
                      os.path.join(self.base_dir,
                                   '240314datY7grV-processed/*.mp4'),
                      os.path.join(self.base_dir,
                                   '231201bmzyrmrN-processed',
                                   'Bariatric surgery',
                                   'Bariatric case */*.mp4'),
                     )
        self.is_vid_map_init = False

    def norm_fn(self, fn):
        bn, _ = os.path.splitext(fn)
        return bn
    
    def get_names(self):
        paths = []
        for g in self.globs:
            paths = paths + glob.glob(g)
        return [osp.basename(p) for p in paths]
    
    def get_video_path(self, video_name):
        if not self.is_vid_map_init:
            self.vid_paths = []
            for g in self.globs:
                self.vid_paths = self.vid_paths + glob.glob(g)
            self.vid_map = {osp.basename(vp):vp for vp in self.vid_paths}
            self.is_vid_map_init = True
        return self.vid_map[video_name]

    def norm_df(self, df):
        return df

class ziyad_test(dataset):
    """
    Dataset to test the example video sent to me by Ziyad.
    """
    def __init__(self):
        self.name = 'ziyad test'
        self.base_dir = '/cluster/projects/madanigroup'
        self.base_dir = osp.join(self.base_dir, 'CLAIM',
                                     'ziyad_comparison',
                                     'Encord downloaded videos')
        self.is_vid_map_init = False
 
    def norm_fn(self, fn):
        bn, _ = osp.splitext(fn)
        return bn
    
    def get_names(self):
        paths = ['Cholec80-Video07-005.mp4']
        return paths    

    def get_video_path(self, video_name):
        if not self.is_vid_map_init:
            self.vid_paths = [osp.join(self.base_dir, video_name)]
            self.vid_map = {osp.basename(video_name): 
                            osp.join(self.base_dir, video_name)}
            self.is_vid_map_init = True
        return self.vid_map[video_name]
    
    def norm_df(self, df):
        return df


class ziyad_tti(dataset):
    """
    Dataset for ziyad's tool tissue interaction project.
    """
    def __init__(self):
        self.name = 'ziyad TTI'
        self.base_dir = '/cluster/projects/madanigroup/CLAIM/subvideos'
        self.globs = (osp.join(self.base_dir, 'Encord_LC_5sec_dataset/*'),
                      osp.join(self.base_dir, 'raw/*'))
        self.is_vid_map_init = False
 
    def norm_fn(self, fn):
        bn, _ = osp.splitext(fn)
        return bn
    
    def get_names(self):
        paths = []
        for g in self.globs:
            paths = paths + glob.glob(g)
        paths = [os.path.basename(p) for p in paths]
        return paths    

    def get_video_path(self, video_name):
        if not self.is_vid_map_init:
            self.vid_paths = []
            for g in self.globs:
                self.vid_paths = self.vid_paths + glob.glob(g)
            self.vid_map = {osp.basename(p): p for p in self.vid_paths}
            self.is_vid_map_init = True
        return self.vid_map[video_name]
    
    def norm_df(self, df):
        return df

