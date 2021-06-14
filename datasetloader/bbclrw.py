"""
This class is currently rather incomplete, only contains the functionality needed
for a first look at the skeletons.
"""

import os

import numpy as np
from tqdm import tqdm

import json
import re

from .datasetloader import DatasetLoader
from .load_lrw_data import *

class BBCLRW(DatasetLoader):
    """
    The Oxford-BBC Lip Reading in the Wild (LRW) Dataset
    https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html
    """
    words = [
        "ABOUT", "ABSOLUTELY", "ABUSE", "ACCESS", "ACCORDING", "ACCUSED", "ACROSS", "ACTION", "ACTUALLY", "AFFAIRS", "AFFECTED", "AFRICA", "AFTER", "AFTERNOON", "AGAIN", "AGAINST", "AGREE", "AGREEMENT", "AHEAD", "ALLEGATIONS", "ALLOW", "ALLOWED", "ALMOST", "ALREADY", "ALWAYS", "AMERICA", "AMERICAN", "AMONG", "AMOUNT", "ANNOUNCED", "ANOTHER", "ANSWER", "ANYTHING", "AREAS", "AROUND", "ARRESTED", "ASKED", "ASKING", "ATTACK", "ATTACKS", "AUTHORITIES", "BANKS", "BECAUSE", "BECOME", "BEFORE", "BEHIND", "BEING", "BELIEVE", "BENEFIT", "BENEFITS", "BETTER", "BETWEEN", "BIGGEST", "BILLION", "BLACK", "BORDER", "BRING", "BRITAIN", "BRITISH", "BROUGHT", "BUDGET", "BUILD", "BUILDING", "BUSINESS", "BUSINESSES", "CALLED", "CAMERON", "CAMPAIGN", "CANCER", "CANNOT", "CAPITAL", "CASES", "CENTRAL", "CERTAINLY", "CHALLENGE", "CHANCE", "CHANGE", "CHANGES", 
        "CHARGE", "CHARGES", "CHIEF", "CHILD", "CHILDREN", "CHINA", "CLAIMS", "CLEAR", "CLOSE", "CLOUD", "COMES", "COMING", "COMMUNITY", "COMPANIES", "COMPANY", "CONCERNS", "CONFERENCE", "CONFLICT", "CONSERVATIVE", "CONTINUE", "CONTROL", "COULD", "COUNCIL", "COUNTRIES", "COUNTRY", "COUPLE", "COURSE", "COURT", "CRIME", "CRISIS", "CURRENT", "CUSTOMERS", "DAVID", "DEATH", "DEBATE", "DECIDED", "DECISION", "DEFICIT", "DEGREES", "DESCRIBED", "DESPITE", "DETAILS", "DIFFERENCE", "DIFFERENT", "DIFFICULT", "DOING", "DURING", "EARLY", "EASTERN", "ECONOMIC", "ECONOMY", "EDITOR", "EDUCATION", "ELECTION", "EMERGENCY", "ENERGY", "ENGLAND", "ENOUGH", "EUROPE", "EUROPEAN", "EVENING", "EVENTS", "EVERY", "EVERYBODY", 
        "EVERYONE", "EVERYTHING", "EVIDENCE", "EXACTLY", "EXAMPLE", "EXPECT", "EXPECTED", "EXTRA", "FACING", "FAMILIES", "FAMILY", "FIGHT", "FIGHTING", "FIGURES", "FINAL", "FINANCIAL", "FIRST", "FOCUS", "FOLLOWING", "FOOTBALL", "FORCE", "FORCES", "FOREIGN", "FORMER", "FORWARD", "FOUND", 
        "FRANCE", "FRENCH", "FRIDAY", "FRONT", "FURTHER", "FUTURE", "GAMES", "GENERAL", "GEORGE", "GERMANY", "GETTING", "GIVEN", "GIVING", "GLOBAL", "GOING", "GOVERNMENT", "GREAT", "GREECE", "GROUND", "GROUP", "GROWING", "GROWTH", "GUILTY", "HAPPEN", "HAPPENED", "HAPPENING", "HAVING", "HEALTH", "HEARD", "HEART", "HEAVY", "HIGHER", "HISTORY", "HOMES", "HOSPITAL", "HOURS", "HOUSE", "HOUSING", "HUMAN", "HUNDREDS", "IMMIGRATION", "IMPACT", "IMPORTANT", "INCREASE", "INDEPENDENT", "INDUSTRY", "INFLATION", "INFORMATION", "INQUIRY", "INSIDE", "INTEREST", "INVESTMENT", "INVOLVED", "IRELAND", "ISLAMIC", "ISSUE", "ISSUES", "ITSELF", "JAMES", "JUDGE", "JUSTICE", "KILLED", "KNOWN", "LABOUR", "LARGE", "LATER", "LATEST", "LEADER", "LEADERS", "LEADERSHIP", "LEAST", "LEAVE", "LEGAL", "LEVEL", "LEVELS", "LIKELY", "LITTLE", "LIVES", "LIVING", "LOCAL", "LONDON", "LONGER", "LOOKING", "MAJOR", "MAJORITY", "MAKES", "MAKING", "MANCHESTER", "MARKET", "MASSIVE", "MATTER", "MAYBE", "MEANS", "MEASURES", "MEDIA", "MEDICAL", "MEETING", "MEMBER", "MEMBERS", "MESSAGE", "MIDDLE", "MIGHT", "MIGRANTS", "MILITARY", "MILLION", "MILLIONS", "MINISTER", "MINISTERS", "MINUTES", "MISSING", "MOMENT", "MONEY", "MONTH", "MONTHS", "MORNING", "MOVING", "MURDER", "NATIONAL", "NEEDS", "NEVER", "NIGHT", "NORTH", "NORTHERN", "NOTHING", "NUMBER", "NUMBERS", "OBAMA", "OFFICE", "OFFICERS", "OFFICIALS", "OFTEN", "OPERATION", "OPPOSITION", 
        "ORDER", "OTHER", "OTHERS", "OUTSIDE", "PARENTS", "PARLIAMENT", "PARTIES", "PARTS", "PARTY", "PATIENTS", "PAYING", "PEOPLE", "PERHAPS", "PERIOD", "PERSON", "PERSONAL", "PHONE", "PLACE", "PLACES", "PLANS", "POINT", "POLICE", "POLICY", "POLITICAL", "POLITICIANS", "POLITICS", "POSITION", "POSSIBLE", "POTENTIAL", "POWER", "POWERS", "PRESIDENT", "PRESS", "PRESSURE", "PRETTY", "PRICE", "PRICES", "PRIME", "PRISON", "PRIVATE", "PROBABLY", "PROBLEM", "PROBLEMS", "PROCESS", "PROTECT", "PROVIDE", "PUBLIC", "QUESTION", "QUESTIONS", "QUITE", "RATES", "RATHER", "REALLY", "REASON", "RECENT", "RECORD", "REFERENDUM", "REMEMBER", "REPORT", "REPORTS", "RESPONSE", "RESULT", "RETURN", "RIGHT", "RIGHTS", "RULES", "RUNNING", "RUSSIA", "RUSSIAN", "SAYING", "SCHOOL", "SCHOOLS", "SCOTLAND", "SCOTTISH", "SECOND", "SECRETARY", "SECTOR", "SECURITY", "SEEMS", "SENIOR", "SENSE", "SERIES", "SERIOUS", "SERVICE", "SERVICES", "SEVEN", "SEVERAL", "SHORT", "SHOULD", "SIDES", "SIGNIFICANT", "SIMPLY", "SINCE", "SINGLE", "SITUATION", "SMALL", "SOCIAL", "SOCIETY", "SOMEONE", "SOMETHING", "SOUTH", "SOUTHERN", "SPEAKING", "SPECIAL", "SPEECH", "SPEND", "SPENDING", "SPENT", "STAFF", "STAGE", "STAND", "START", "STARTED", "STATE", "STATEMENT", "STATES", "STILL", "STORY", "STREET", "STRONG", "SUNDAY", "SUNSHINE", "SUPPORT", "SYRIA", "SYRIAN", "SYSTEM", "TAKEN", "TAKING", "TALKING", "TALKS", "TEMPERATURES", "TERMS", "THEIR", 
        "THEMSELVES", "THERE", "THESE", "THING", "THINGS", "THINK", "THIRD", "THOSE", "THOUGHT", "THOUSANDS", "THREAT", "THREE", "THROUGH", "TIMES", "TODAY", "TOGETHER", "TOMORROW", "TONIGHT", "TOWARDS", "TRADE", "TRIAL", "TRUST", "TRYING", "UNDER", "UNDERSTAND", "UNION", "UNITED", "UNTIL", "USING", "VICTIMS", "VIOLENCE", "VOTERS", "WAITING", "WALES", "WANTED", "WANTS", "WARNING", "WATCHING", "WATER", "WEAPONS", "WEATHER", "WEEKEND", "WEEKS", "WELCOME", "WELFARE", "WESTERN", "WESTMINSTER", "WHERE", "WHETHER", "WHICH", "WHILE", "WHOLE", "WINDS", "WITHIN", "WITHOUT", "WOMEN", "WORDS", "WORKERS", "WORKING", "WORLD", "WORST", "WOULD", "WRONG", "YEARS", "YESTERDAY", "YOUNG"
    ]

    landmarks = [
        "point-1","point-2","point-3","point-4","point-5","point-6","point-7","point-8","point-9","point-10",
        "point-11","point-12","point-13","point-14","point-15","point-16","point-17","point-18","point-19","point-20",
        "point-21","point-22","point-23","point-24","point-25","point-26","point-27","point-28","point-29","point-30",
        "point-31","point-32","point-33","point-34","point-35","point-36","point-37","point-38","point-39","point-40",
        "point-41","point-42","point-43","point-44","point-45","point-46","point-47","point-48","point-49","point-50",
        "point-51","point-52","point-53","point-54","point-55","point-56","point-57","point-58","point-59","point-60",
        "point-61","point-62","point-63","point-64","point-65","point-66","point-67","point-68"
    ]
    splits = ["default"]

    def __init__(self, base_dir, lazy_loading=True):
        """
        Parameters
        ----------
        base_dir : string
            folder with dataset on disk
        lazy_loading : bool, optional (default is True)
            Only load individual data items when queried
        """
        self._data_cols = [
            "keypoint-filename",
            "keypoints3D",
            "keypoints2D",
            # "video-filename",
        ]
        self._data = {
            "keypoint-filename": [],
            # The dataset also contains these, to be implemented if/when needed
            # "video-filename": [],
        }

        # describe the dataset split, containing the ids of elements in the
        # respective sets
        self._splits = {
            split: {
                "train": [],
                "val": [],
                "test": []
            }
            for split in BBCLRW.splits
        }

        self._length = 0
        #i_data = "/cache/lrw/lipread_landmarks/dlib68_2d_sparse_json/lipread_mp4"
        # or i_data = "/cache/lrw/lipread_landmarks/dlib68_2d_sparse_json_defects_not_one_face/lipread_mp4"
        selected_n_classes = 10 # the max is 500

        for cls_id, cls in tqdm(enumerate(BBCLRW.words)):
            # load dat for this class
            if cls_id >= selected_n_classes:
                break
            for split in self._splits["default"].keys():
                d = os.path.join(base_dir, cls, split)
                for _, _, files in os.walk(d):
                    for filename in files:
                        if filename.endswith('.json'):
                            self._data["keypoint-filename"].append(os.path.join(d, filename))
                            self._splits["default"][split].append(self._length)
                            self._length += 1

        super().__init__(lazy_loading)

    

    def load_keypointfile(self, filename):
        """
        Load the keypoints sequence from the given file.

        For reference, the format of the skeleton-file is:
        num_frames
        num_bodies (in frame 0)
        body_0_info: ID, clipped_edges, hand_left_confidence, hand_left_state,hand_right_confidence, hand_right_state, is_restricted, lean_x, lean_y, tracking_state
        num_joints (of person 0 in frame 0, constant 25)
        joint_0_info: x, y, z, depth_x, depth_y, rgb_x, rgb_y, orientation_w, orientation_x, orientation_y, orientation_z, tracking_state
        . . .
        body_1_info...
        . . .
        num_bodies (in frame 1)
        . . .

        Parameters
        ----------
        filename : string
            Filename of the file containing a skeleton sequence
        """
        face_rect_array, landmarks_array, duration_array = load_one_json_file(filename, isDebug=False)
        num_frames = landmarks_array.shape[0]
        if "keypoints2D" in self._selected_cols:
            persons2d = np.zeros((1, num_frames, 68, 2))
            persons2d[0] = landmarks_array
        if "keypoints3D" in self._selected_cols:
            persons3d = np.zeros((1, num_frames, 68, 3))
            persons3d[0] = landmarks_array
                    
        persons = []
        if "keypoints2D" in self._selected_cols:
            persons.append(persons2d)
        if "keypoints3D" in self._selected_cols:
            persons.append(persons3d)
        return persons

    def __getitem__(self, index):
        """
        Indexing access to the dataset.

        Returns a dictionary of all currently selected data columns of the
        selected item.
        """
        data = super().__getitem__(index)
        # super() provides all non-lazy access, only need to do more for data
        # that hasn't been loaded previously
        if len(self._selected_cols - data.keys()) > 0:
            if ("keypoints3D" in self._selected_cols
                    or "keypoints2D" in self._selected_cols):
                keypoints = self.load_keypointfile(
                    self._data["keypoint-filename"][index])
                if "keypoints2D" in self._selected_cols:
                    data["keypoints2D"] = keypoints.pop()
                if "keypoints3D" in self._selected_cols:
                    data["keypoints3D"] = keypoints.pop()
        return data