from .preprocess import fetch_lfw_people
import tqdm
import numpy as np
from .helper import identity
from .augment import augment_data
import matplotlib.pyplot as plt
from PIL import Image


def export_dataset_objects(
    min_faces: int | None = 20,
    max_faces: int | None = None,
    shuffle=True,
    random_state=None,
    verbose=True,
    augment=True,
    desired_shape: tuple[int, int] = (250, 250),
    augmentation_count: int = 10,
    augmentation_pipeline=None,
):
    X, Y = fetch_lfw_people(
        min_faces=min_faces,
        max_faces=max_faces,
        shuffle=shuffle,
        random_state=random_state,
        verbose=verbose,
    )

    tracker = tqdm.tqdm if verbose else identity

    __x = []
    for x in tracker(X):
        __x.append(x())
    # exit(1)

    if augment:
        xy = []
        for i, x in tracker(enumerate(__x)):
            aug_data = augment_data(
                x,
                desired_shape=desired_shape,
                augmentation_count=augmentation_count,
                augmentation_pipeline=augmentation_pipeline,
            )

            for data in aug_data:
                data = Image.fromarray(data).convert("L")
                data = np.array(data)
                xy.append([data, Y[i]])
            # print(xy)
            # fig, axes = plt.subplots(2, 5, figsize=(20, 10))
            # axes = axes.flatten()

            # for img, ax in zip(aug_data, axes):
            #     ax.imshow(img)
            #     ax.axis("off")
            # plt.tight_layout()
            # plt.show()

            # exit(1)
        x_data = np.array(xy)[:, 0]
        y_data = np.array(xy)[:, 1]
        x_data.dump("Dataset/x.npy")
        y_data.dump("Dataset/y.npy")
    else:
        x_data = np.array(__x)
        x_data.dump("Dataset/x.npy")
        Y.dump("Dataset/y.npy")


if __name__ == "__main__":
    export_dataset_objects(shuffle=False, min_faces=60, augment=False)
