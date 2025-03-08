import json
import numpy as np
from sklearn.model_selection import MultilabelStratifiedShuffleSplit

def split_dataset(json_path):
    # JSON 파일 읽기
    with open(json_path, 'r') as f:
        data = json.load(f)

    appearance_mapping = data["meta_info"]["appearance_mapping"]
    action_mapping = data["meta_info"]["action_mapping"]
    clips = data["clips"]

    # 클립 ID와 레이블 수집
    clip_ids = []
    appearance_labels = []
    action_labels = []
    for clip_id, clip_info in clips.items():
        appearance_attr = clip_info["attributes"]["appearance"]
        action_attr = clip_info["attributes"]["action"]
        clip_ids.append(clip_id)
        appearance_labels.append(appearance_attr)
        action_labels.append(action_attr)

    clip_ids = np.array(clip_ids)
    appearance_labels = np.array(appearance_labels)
    action_labels = np.array(action_labels)

    # appearance와 action 레이블을 결합하여 다중 레이블로 처리
    combined_labels = np.concatenate([appearance_labels, action_labels], axis=1)

    # MultilabelStratifiedShuffleSplit을 사용하여 train/valid/test로 나누기
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(splitter.split(clip_ids, combined_labels))

    # validation과 test로 temp set을 50%씩 나누기
    val_splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(val_splitter.split(clip_ids[temp_idx], combined_labels[temp_idx]))

    return train_idx, temp_idx[val_idx], temp_idx[test_idx]