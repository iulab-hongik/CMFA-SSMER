import copy
import random
import logging
import json

import cv2
import numpy
import torch
import numpy as np
import os
import utils

from torch.utils.data import Dataset

from utils import generate_target


logger = logging.getLogger(__name__)


class ESIE_Dataset(Dataset):
    def __init__(self, cfg, root, is_train='test', transform=None):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.is_train = is_train
        self.root = cfg.ESIE.ROOT
        self.number_landmarks = cfg.ESIE.NUM_POINT
        self.flip_index = np.genfromtxt("Mirror_esie.txt", dtype=int, delimiter=',')

        self.Fraction = cfg.ESIE.FRACTION
        self.Translation_Factor = cfg.ESIE.TRANSLATION
        self.Rotation_Factor = cfg.ESIE.ROTATION
        self.Scale_Factor = cfg.ESIE.SCALE
        self.Occlusion_Mean = cfg.ESIE.OCCLUSION_MEAN
        self.Occlusion_Std = cfg.ESIE.OCCLUSION_STD
        self.Flip = cfg.ESIE.FLIP
        self.Occlusion = cfg.ESIE.OCCLUSION
        self.Transfer = cfg.ESIE.CHANNEL_TRANSFER

        self.Heatmap_size = cfg.MODEL.HEATMAP

        self.Data_Format = cfg.ESIE.DATA_FORMAT
        self.Event_Path = os.path.join(cfg.ESIE.ROOT, cfg.DATASET.REPR)
        self.Event_Repr = cfg.DATASET.REPR

        self.Transform = transform
        self.annotation_dir = os.path.join(self.root, 'label')

        self.database = self.get_file_information()

    def get_file_information(self):
        Data_base = []
        txt_files = []
        for root, _, files in os.walk(self.annotation_dir):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        for file_path in txt_files:
            file_name = os.path.basename(file_path).split('.')[0]
            subuser = os.path.basename(os.path.dirname(file_path))
            user = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
            img_name = file_name + '.png'

            with open(file_path) as f:
                temp_info = f.read().splitlines()
            temp_info = temp_info[0].split(' ')

            max_index = np.array([float(temp_info[12]), float(temp_info[13])])
            min_index = np.array([float(temp_info[10]), float(temp_info[11])])
            temp_point = [float(temp_info[i]) for i in range(0, 10)]  # landmakrs
            point_coord = np.array(temp_point, dtype=float).reshape(self.number_landmarks, 2)

            temp_box = np.array([min_index[0], min_index[1], max_index[0] - min_index[0],
                                 max_index[1] - min_index[1]])  # bbox : x, y, w, h

            img_path = os.path.join(self.root, "rgb", user, subuser, img_name)
            event_path = os.path.join(self.Event_Path, user, subuser, img_name) # repr 절대경로
            Data_base.append({'Img': img_path,
                              'Event': event_path,
                              'bbox': temp_box,
                              'point': point_coord,
                              'user': user,
                              'subuser': subuser})
        return Data_base

    def Image_Flip(self, Img, GT):
        Mirror_GT = []
        width = Img.shape[1]
        for i in self.flip_index:
            Mirror_GT.append([width - 1 - GT[i][0], GT[i][1]])
        Img = cv2.flip(Img, 1)
        return Img, numpy.array(Mirror_GT)

    def Channel_Transfer(self, Img, Flag):
        if Flag == 1:
            Img = cv2.cvtColor(Img, cv2.COLOR_RGB2GRAY)
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        return Img

    def Create_Occlusion(self, Img):
        Occlusion_width = int(self.Image_size * np.random.normal(self.Occlusion_Mean, self.Occlusion_Std))
        Occlusion_high = int(self.Image_size * np.random.normal(self.Occlusion_Mean, self.Occlusion_Std))
        Occlusion_x = np.random.randint(0, self.Image_size - Occlusion_width)
        Occlusion_y = np.random.randint(0, self.Image_size - Occlusion_high)

        Img[Occlusion_y:Occlusion_y + Occlusion_high, Occlusion_x:Occlusion_x + Occlusion_width, 0] = \
            np.random.randint(0, 256)
        Img[Occlusion_y:Occlusion_y + Occlusion_high, Occlusion_x:Occlusion_x + Occlusion_width, 1] = \
            np.random.randint(0, 256)
        Img[Occlusion_y:Occlusion_y + Occlusion_high, Occlusion_x:Occlusion_x + Occlusion_width, 2] = \
            np.random.randint(0, 256)

        return Img

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        db_slic = copy.deepcopy(self.database[idx])
        Img_path = db_slic['Img']
        Event_path = db_slic['Event']
        BBox = db_slic['bbox']
        Points = db_slic['point']
        Annotated_Points = Points.copy()
        Points_ev = Points.copy()
        User = db_slic['user']
        Subuser = db_slic['subuser']

        Img = cv2.imread(Img_path)
        event_Img = cv2.imread(Event_path)
        Img_shape = Img.shape
        event_shape = event_Img.shape

        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)
        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)

        event_Img = cv2.cvtColor(event_Img, cv2.COLOR_RGB2BGR)
        if len(event_shape) < 3:
            event_Img = cv2.cvtColor(event_Img, cv2.COLOR_GRAY2RGB)
        else:
            if event_shape[2] == 4:
                event_Img = cv2.cvtColor(event_Img, cv2.COLOR_RGBA2RGB)
            elif event_shape[2] == 1:
                event_Img = cv2.cvtColor(event_Img, cv2.COLOR_GRAY2RGB)

        if self.is_train == 'train':
            Rotation_Factor = self.Rotation_Factor * np.pi / 180.0
            Scale_Factor = self.Scale_Factor
            Translation_X_Factor = self.Translation_Factor
            Translation_Y_Factor = self.Translation_Factor

            angle = np.clip(np.random.normal(0, Rotation_Factor), -2 * Rotation_Factor, 2 * Rotation_Factor)
            Scale = np.clip(np.random.normal(self.Fraction, Scale_Factor), self.Fraction - Scale_Factor, self.Fraction + Scale_Factor)

            Translation_X = np.clip(np.random.normal(0, Translation_X_Factor), -Translation_X_Factor, Translation_X_Factor)
            Translation_Y = np.clip(np.random.normal(0, Translation_Y_Factor), -Translation_Y_Factor, Translation_Y_Factor)

            trans = utils.get_transforms(BBox, Scale, angle, self.Image_size, shift_factor=[Translation_X, Translation_Y])

            input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)
            event_input = cv2.warpAffine(event_Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

            for i in range(self.number_landmarks):
                Points[i,0:2] = utils.affine_transform(Points[i,0:2], trans)
                Points_ev[i,0:2] = utils.affine_transform(Points[i,0:2], trans)

            if self.Flip is True:
                Flip_Flag = np.random.randint(0, 2)
                if Flip_Flag == 1:
                    event_input, Points = self.Image_Flip(event_input, Points)
                    event_input, _ = self.Image_Flip(event_input, Points_ev)

            if self.Transfer is True:
                Transfer_Flag = np.random.randint(0, 5)
                input = self.Channel_Transfer(input, Transfer_Flag)
                event_input = self.Channel_Transfer(event_input, Transfer_Flag)

            if self.Occlusion is True:
                Occlusion_Flag = np.random.randint(0, 2)
                if Occlusion_Flag == 1:
                    input = self.Create_Occlusion(input)
                    event_input = self.Create_Occlusion(event_input)

            if self.Transform is not None:
                input = self.Transform(input)
                event_input = self.Transform(event_input)



            meta = {'Event_path': Event_path,
                    'Points': Points / (self.Image_size),
                    'BBox': BBox,
                    'trans': trans,
                    'Scale': Scale,
                    'angle': angle,
                    'Translation': [Translation_X, Translation_Y],
                    'User': User,
                    'Subuser': Subuser
                    }
            return input, event_input, meta

        else:
            trans = utils.get_transforms(BBox, self.Fraction, 0.0, self.Image_size, shift_factor=[0.0, 0.0])

            input = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)
            event_input = cv2.warpAffine(event_Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)

            for i in range(self.number_landmarks):
                Points[i, 0:2] = utils.affine_transform(Points[i, 0:2], trans)

            meta = {
                "Annotated_Points": Annotated_Points,
                'Event_path': Event_path,
                'Points': Points / (self.Image_size),
                'BBox': BBox,
                'trans': trans,
                'Scale': self.Fraction,
                'angle': 0.0,
                'Translation': [0.0, 0.0],
                'User': User,
                'Subuser': Subuser
            }

            if self.Transform is not None:
                input = self.Transform(input)
                event_input = self.Transform(event_input)

            return input, event_input, meta
