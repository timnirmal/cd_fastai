from pathlib import Path

from fastai.data.block import DataBlock, CategoryBlock
from fastai.data.transforms import get_image_files, GrandparentSplitter, parent_label
from fastai.vision.augment import Resize
from fastai.vision.data import ImageBlock


def get_data_loaders_fastai(train_dir, valid_dir):
    train_dir = Path(train_dir)
    valid_dir = Path(valid_dir)

    path = Path(train_dir).parent  # Assuming train and valid directories share the same parent
    print(path)

    datablock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=GrandparentSplitter(train_name=train_dir.name, valid_name=valid_dir.name),
        get_y=parent_label,
        item_tfms=Resize(224),
        # batch_tfms=aug_transforms(size=224, min_scale=0.75)
    )

    dls = datablock.dataloaders(path, bs=64, num_workers=0)

    return dls
