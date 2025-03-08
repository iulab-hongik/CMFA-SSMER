import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from simsiam.loader import GaussianBlur
from sklearn.model_selection import train_test_split
import json


class MultiEventDataset(Dataset):
    def __init__(self, root_dir, representations):
        self.root_dir = root_dir

        #### check representation folder validation ####
        self.representation_dir = []
        for representation in representations:
            self.representation_dir.append(os.path.join(root_dir, representation))
        self.representation_folders = []
        for representation_dir in self.representation_dir:
            self.representation_folders.append([d for d in os.listdir(representation_dir) if os.path.isdir(os.path.join(representation_dir, d))])


        self.video_folders = set(self.representation_folders[0])
        for representation_folder in self.representation_folders[1:]:
            self.video_folders.intersection_update(representation_folder)

        self.missing_folders = set()
        for representation_folder in self.representation_folders:
            self.missing_folders.update(set(representation_folder) - self.video_folders)
        print("Used folders", len(self.video_folders))
        print("Missing folders", len(self.missing_folders))
        self.video_folders = list(self.video_folders)
        self.missing_folders = list(self.missing_folders)
        ################################################

        # transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        self.transform = transforms.Compose(augmentation)

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        representations = random.sample(self.representation_dir, 2)
        video_folder_path1 = os.path.join(representations[0], video_folder)
        video_folder_path2 = os.path.join(representations[1], video_folder)

        event_images1 = sorted(os.listdir(video_folder_path1))
        event_images2 = sorted(os.listdir(video_folder_path2))

        image_number = min(len(event_images1), len(event_images2))
        if image_number < 2:
            raise ValueError(f"폴더 {video_folder}에 이미지가 충분하지 않습니다.")
        number = random.randint(0, image_number-1)

        # 같은 폴더에서 무작위로 두 개의 이미지 선택
        img1_name = event_images1[number]
        img2_name = event_images2[number]

        img1 = Image.open(os.path.join(video_folder_path1, img1_name)).convert('RGB')
        img2 = Image.open(os.path.join(video_folder_path2, img2_name)).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2


class EventDatasetWithSplit(Dataset):
    def __init__(self, root_dir, representations, split='train', test_size=0.2, val_size=0.1, random_state=42, split_path='./data/train_val_test_split_multilabel.json'):
        self.root_dir = root_dir
        self.representations = representations
        self.split = split  # 'train', 'validation', 'test'
        self.test_size = test_size
        self.val_size = val_size

        #### check representation folder validation ####
        self.representation_dir = []
        for representation in representations:
            self.representation_dir.append(os.path.join(root_dir, representation))

        self.representation_folders = []
        for representation_dir in self.representation_dir:
            self.representation_folders.append(
                [d for d in os.listdir(representation_dir) if os.path.isdir(os.path.join(representation_dir, d))])

        self.video_folders = set(self.representation_folders[0])
        for representation_folder in self.representation_folders[1:]:
            self.video_folders.intersection_update(representation_folder)

        self.video_folders = list(self.video_folders)


        if split == 'train':
            self.video_folders = self._read_split(split_path)['train']
        elif split == 'validation':
            self.video_folders = self._read_split(split_path)['validation']
        elif split == 'test':
            self.video_folders = self._read_split(split_path)['test']
        else:
            print("Split type is not support.")
            exit()
        ################################################

        # transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        self.transform = transforms.Compose(augmentation)

    def _read_split(self, split_path):
        with open(split_path, 'r') as f:
            data = json.load(f)

        data_list = data['train'] + data['validation'] + data['test']
        for video_folder in self.video_folders:
            if video_folder not in data_list:
                print(video_folder, 'not in split json file')
                exit()

        return data

    def _split_dataset(self, random_state):
        # 전체 데이터셋에서 train, validation, test로 나누는 메소드
        idx_list = list(range(len(self.video_folders)))

        train_idx, temp_idx = train_test_split(
            idx_list, test_size=self.test_size, random_state=random_state
        )

        val_size_adjusted = self.val_size / self.test_size  # test 데이터를 제외한 비율로 조정
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=val_size_adjusted, random_state=random_state
        )

        return train_idx, val_idx, test_idx

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        video_folder_path = os.path.join(self.representation_dir[0], video_folder)

        event_images = sorted(os.listdir(video_folder_path))

        image_number = len(event_images)
        number = random.randint(0, image_number-1)

        img1_name = event_images[number]
        img2_name = event_images[number]

        img1 = Image.open(os.path.join(video_folder_path, img1_name)).convert('RGB')
        img2 = Image.open(os.path.join(video_folder_path, img2_name)).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2


class DoubleEventDatasetWithSplit(Dataset):
    def __init__(self, root_dir, representations, split='train', test_size=0.2, val_size=0.1, random_state=42, split_path='./data/train_val_test_split_multilabel.json'):
        self.root_dir = root_dir
        self.representations = representations
        self.split = split  # 'train', 'validation', 'test'
        self.test_size = test_size
        self.val_size = val_size

        #### check representation folder validation ####
        self.representation_dir = []
        for representation in representations:
            self.representation_dir.append(os.path.join(root_dir, representation))

        self.representation_folders = []
        for representation_dir in self.representation_dir:
            self.representation_folders.append(
                [d for d in os.listdir(representation_dir) if os.path.isdir(os.path.join(representation_dir, d))])

        self.video_folders = set(self.representation_folders[0])
        for representation_folder in self.representation_folders[1:]:
            self.video_folders.intersection_update(representation_folder)

        self.video_folders = list(self.video_folders)

        with open(os.path.join(root_dir, 'train_val_test_split_multilabel.json'), 'r') as splits_file:
            splits = json.load(splits_file)
            if split == 'train':
                self.video_folders = splits['train']
            elif split == 'val':
                self.video_folders = splits['validation']
            elif split == 'test':
                self.video_folders = splits['test']
            else:
                raise ValueError(f"Unknown split value:{split} Please use one of 'train', 'val', or 'test'.")

        ################################################

        # transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        self.transform = transforms.Compose(augmentation)

    def _split_dataset(self, random_state):
        # 전체 데이터셋에서 train, validation, test로 나누는 메소드
        idx_list = list(range(len(self.video_folders)))

        train_idx, temp_idx = train_test_split(
            idx_list, test_size=self.test_size, random_state=random_state
        )

        val_size_adjusted = self.val_size / self.test_size  # test 데이터를 제외한 비율로 조정
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=val_size_adjusted, random_state=random_state
        )

        return train_idx, val_idx, test_idx

    def _read_split(self, split_path):
        with open(split_path, 'r') as f:
            data = json.load(f)

        data_list = data['train'] + data['validation'] + data['test']
        for video_folder in self.video_folders:
            if video_folder not in data_list:
                print(video_folder, 'not in split json file')
                exit()

        return data

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        video_folder_path1 = os.path.join(self.representation_dir[0], video_folder)
        video_folder_path2 = os.path.join(self.representation_dir[1], video_folder)

        event_images1 = sorted(os.listdir(video_folder_path1))
        event_images2 = sorted(os.listdir(video_folder_path2))

        # if len(event_images1) != len(event_images2):
        #     print("The number of videos is not same.", len(event_images1), len(event_images2))
        image_number = min(len(event_images1), len(event_images2))
        number = random.randint(0, image_number-1)

        img1_name = event_images1[number]
        img2_name = event_images2[number]

        img1 = Image.open(os.path.join(video_folder_path1, img1_name)).convert('RGB')
        img2 = Image.open(os.path.join(video_folder_path2, img2_name)).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2


class MultiEventDatasetWithSplit(Dataset):
    def __init__(self, root_dir, representations, split='train', test_size=0.2, val_size=0.1, random_state=42, split_path='./data/train_val_test_split_multilabel.json'):
        self.root_dir = root_dir
        self.representations = representations
        self.split = split  # 'train', 'validation', 'test'
        self.test_size = test_size
        self.val_size = val_size

        #### check representation folder validation ####
        self.representation_dir = []
        for representation in representations:
            self.representation_dir.append(os.path.join(root_dir, representation))

        self.representation_folders = []
        for representation_dir in self.representation_dir:
            self.representation_folders.append(
                [d for d in os.listdir(representation_dir) if os.path.isdir(os.path.join(representation_dir, d))])

        # removes the items that is not in all representations
        self.video_folders = set(self.representation_folders[0])
        for representation_folder in self.representation_folders[1:]:
            self.video_folders.intersection_update(representation_folder)

        self.video_folders = list(self.video_folders)

        if split == 'train':
            self.video_folders = self._read_split(split_path)['train']
        elif split == 'validation':
            self.video_folders = self._read_split(split_path)['validation']
        elif split == 'test':
            self.video_folders = self._read_split(split_path)['test']
        else:
            print("Split type is not support.")
            exit()
        ################################################

        # transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        self.transform = transforms.Compose(augmentation)

    def _split_dataset(self, random_state):
        # 전체 데이터셋에서 train, validation, test로 나누는 메소드
        idx_list = list(range(len(self.video_folders)))

        train_idx, temp_idx = train_test_split(
            idx_list, test_size=self.test_size, random_state=random_state
        )

        val_size_adjusted = self.val_size / self.test_size  # test 데이터를 제외한 비율로 조정
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=val_size_adjusted, random_state=random_state
        )

        return train_idx, val_idx, test_idx

    def _read_split(self, split_path):
        with open(split_path, 'r') as f:
            data = json.load(f)

        data_list = data['train'] + data['validation'] + data['test']
        for video_folder in self.video_folders:
            if video_folder not in data_list:
                print(video_folder, 'not in split json file')
                exit()

        return data


    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        video_folder_path1 = os.path.join(self.representation_dir[0], video_folder)
        video_folder_path2 = os.path.join(self.representation_dir[1], video_folder)
        video_folder_path3 = os.path.join(self.representation_dir[2], video_folder)

        event_images1 = sorted(os.listdir(video_folder_path1))
        event_images2 = sorted(os.listdir(video_folder_path2))
        event_images3 = sorted(os.listdir(video_folder_path3))

        if len(event_images1) != len(event_images2) or len(event_images1) != len(event_images3):
            print("The number of videos is not same.")
            exit()
        image_number = len(event_images1)
        number = random.randint(0, image_number-1)

        img1_name = event_images1[number]
        img2_name = event_images2[number]
        img3_name = event_images3[number]

        img1 = Image.open(os.path.join(video_folder_path1, img1_name)).convert('RGB')
        img2 = Image.open(os.path.join(video_folder_path2, img2_name)).convert('RGB')
        img3 = Image.open(os.path.join(video_folder_path3, img3_name)).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3


class MaxEventDatasetWithSplit(Dataset):
    def __init__(self, root_dir, representations, split='train', test_size=0.2, val_size=0.1, random_state=42):
        self.root_dir = root_dir
        self.representations = representations
        self.split = split  # 'train', 'val', 'test'
        self.test_size = test_size
        self.val_size = val_size

        #### check representation folder validation ####
        self.representation_dir = []
        for representation in representations:
            self.representation_dir.append(os.path.join(root_dir, representation))

        self.representation_folders = []
        for representation_dir in self.representation_dir:
            self.representation_folders.append(
                [d for d in os.listdir(representation_dir) if os.path.isdir(os.path.join(representation_dir, d))])

        # self.video_folders = list(self.video_folders)
        with open(os.path.join(root_dir, 'train_val_test_split_multilabel.json'), 'r') as splits_file:
            splits = json.load(splits_file)
            if split == 'train':
                self.video_folders = splits['train']
            elif split == 'val':
                self.video_folders = splits['validation']
            elif split == 'test':
                self.video_folders = splits['test']
            else:
                raise ValueError(f"Unknown split value:{split} Please use one of 'train', 'val', or 'test'.")
        ################################################

        # transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        self.transform = transforms.Compose(augmentation)

    def _read_split(self, split_path):
        with open(split_path, 'r') as f:
            data = json.load(f)

        data_list = data['train'] + data['validation'] + data['test']
        for video_folder in self.video_folders:
            if video_folder not in data_list:
                print(video_folder, 'not in split json file')
                exit()

        return data

    def _split_dataset(self, random_state):
        # 전체 데이터셋에서 train, validation, test로 나누는 메소드
        idx_list = list(range(len(self.video_folders)))

        train_idx, temp_idx = train_test_split(
            idx_list, test_size=self.test_size, random_state=random_state
        )

        val_size_adjusted = self.val_size / self.test_size  # test 데이터를 제외한 비율로 조정
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=val_size_adjusted, random_state=random_state
        )

        return train_idx, val_idx, test_idx

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        video_folder_path = os.path.join(self.representation_dir[0], video_folder)

        event_images = sorted(os.listdir(video_folder_path))

        image_number = len(event_images)
        number = random.randint(0, image_number-1)

        img1_name = event_images[number]
        img2_name = event_images[number]

        img1 = Image.open(os.path.join(video_folder_path, img1_name)).convert('RGB')
        img2 = Image.open(os.path.join(video_folder_path, img2_name)).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2

class MaxMultiEventDatasetWithSplit(Dataset):
    def __init__(self, root_dir, representations, split='train', test_size=0.2, val_size=0.1):
        self.root_dir = root_dir
        self.representations = representations
        self.split = split  # 'train', 'validation', 'test'
        self.test_size = test_size
        self.val_size = val_size

        #### check representation folder validation ####
        self.representation_dir = []
        for representation in representations:
            self.representation_dir.append(os.path.join(root_dir, representation))

        self.representation_folders = []
        for representation_dir in self.representation_dir:
            self.representation_folders.append(
                [d for d in os.listdir(representation_dir) if os.path.isdir(os.path.join(representation_dir, d))])


        with open(os.path.join(root_dir, 'train_val_test_split_multilabel.json'), 'r') as splits_file:
            splits = json.load(splits_file)
            if split == 'train':
                self.video_folders = splits['train']
            elif split == 'val':
                self.video_folders = splits['validation']
            elif split == 'test':
                self.video_folders = splits['test']
            else:
                raise ValueError(f"Unknown split value:{split} Please use one of 'train', 'val', or 'test'.")
        ################################################

        # transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        self.transform = transforms.Compose(augmentation)

    def _split_dataset(self, random_state):
        # 전체 데이터셋에서 train, validation, test로 나누는 메소드
        idx_list = list(range(len(self.video_folders)))

        train_idx, temp_idx = train_test_split(
            idx_list, test_size=self.test_size, random_state=random_state
        )

        val_size_adjusted = self.val_size / self.test_size  # test 데이터를 제외한 비율로 조정
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=val_size_adjusted, random_state=random_state
        )

        return train_idx, val_idx, test_idx

    def _read_split(self, split_path):
        with open(split_path, 'r') as f:
            data = json.load(f)

        data_list = data['train'] + data['validation'] + data['test']
        for video_folder in self.video_folders:
            if video_folder not in data_list:
                print(video_folder, 'not in split json file')
                exit()

        return data


    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        video_folder_path1 = os.path.join(self.representation_dir[0], video_folder)
        video_folder_path2 = os.path.join(self.representation_dir[1], video_folder)
        video_folder_path3 = os.path.join(self.representation_dir[2], video_folder)

        event_images1 = sorted(os.listdir(video_folder_path1))
        event_images2 = sorted(os.listdir(video_folder_path2))
        event_images3 = sorted(os.listdir(video_folder_path3))

        if len(event_images1) != len(event_images2) or len(event_images1) != len(event_images3):
            # print("The number of videos is not same. You're using the min number of videos.")
            image_number = min(len(event_images1), len(event_images2), len(event_images3))
        else:
            image_number = len(event_images1)
        number = random.randint(0, image_number-1)

        img1_name = event_images1[number]
        img2_name = event_images2[number]
        img3_name = event_images3[number]

        img1 = Image.open(os.path.join(video_folder_path1, img1_name)).convert('RGB')
        img2 = Image.open(os.path.join(video_folder_path2, img2_name)).convert('RGB')
        img3 = Image.open(os.path.join(video_folder_path3, img3_name)).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3