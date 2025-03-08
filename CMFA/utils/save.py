import os
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from Config import cfg
from Config import update_config


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def save_img(meta, landmarks, radius=2, color='lime'):
    dest_dir = f"{cfg.TEST.RESULT_PATH}/{cfg.DATASET.DATASET}/{cfg.MODEL.NAME}/results"
    createDirectory(dest_dir)
    img_path = meta['Event_path'][0]
    img_name = img_path.split('/')[-1]

    # Open the txt file
    image = Image.open(img_path)
    if cfg.DATASET.DATASET == 'eCelebV_v2e':
        image = image.resize((512, 512))
    draw = ImageDraw.Draw(image)

    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.cpu().numpy()
        landmarks = landmarks[0]

    if cfg.DATASET.DATASET in ['ESIE']:
        for i, point in enumerate(landmarks):
            x, y = point
            if i == 3 or i == 4:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='yellow', outline='yellow')
            else:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=color)
    else:
        for i, point in enumerate(landmarks):
            x, y = point
            if i == 96 or i == 97:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red', outline='red')
            else:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=color)
    save_dir = os.path.join(dest_dir, img_name)
    image.save(save_dir)


def save_comparison(meta, stage1=None, stage2=None, stage3=None, gt=None):
    dest_dir = f"{cfg.TEST.RESULT_PATH}/{cfg.DATASET.DATASET}/{cfg.MODEL.NAME}/comparison"
    createDirectory(dest_dir)
    img_path = meta['Event_path'][0]
    img_name = img_path.split('/')[-1]
    image = Image.open(img_path)

    if cfg.DATASET.DATASET == 'eCelebV_v2e':
        save_path = os.path.join(dest_dir, img_name)
        image = image.resize((512, 512))

    elif cfg.DATASET.DATASET in ['ESIE']:
        user = meta['User'][0]
        subuser = meta['Subuser'][0]
        save_path = os.path.join(dest_dir, f'{user}_{subuser}_{img_name}')


    # Handle grayscale images dynamically
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Convert grayscale to RGB for consistent processing

    # Plot the image, predictions, and ground truth
    plot_stages = [("Original Image", None), ("Stage 1 Prediction", stage1)]
    if stage2 is not None:
        plot_stages.append(("Stage 2 Prediction", stage2))
    if stage3 is not None:
        plot_stages.append(("Stage 3 Prediction", stage3))
    plot_stages.append(("Ground Truth", gt))

    # Create subplots dynamically
    fig, axes = plt.subplots(1, len(plot_stages), figsize=(5 * len(plot_stages), 5))

    # Plot each stage
    for ax, (title, stage) in zip(axes, plot_stages):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

        if stage is not None:
            plot_landmarks_on_image(image, stage, ax, title)

    # Save and close figure
    plt.tight_layout(pad=2.0)
    plt.savefig(save_path)
    plt.close()

    # print(f"Comparison saved at: {save_path}")


def plot_landmarks_on_image(image, landmarks, axis, title, color='lime'):
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    radius = 2
    if cfg.DATASET.DATASET in ['ESIE']:
        for i, point in enumerate(landmarks):
            x, y = point
            if i == 3 or i == 4:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='yellow', outline='yellow')
            else:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=color)
    else:
        for i, point in enumerate(landmarks):
            x, y = point
            if i == 96 or i == 97:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red', outline='red')
            else:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=color)
    axis.imshow(img_copy)
    axis.set_title(title)
    axis.axis("off")