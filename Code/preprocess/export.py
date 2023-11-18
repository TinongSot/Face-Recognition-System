from sklearn.model_selection import train_test_split
from .preprocess import fetch_lfw_people
import tqdm
import numpy as np
from .helper import identity
from .augment import augment_data
import matplotlib.pyplot as plt
from PIL import Image
from rich import print

# TODO: Documentation


def export_dataset_objects(
    min_faces: int | None = 20,
    max_faces: int | None = None,
    hard_limit: bool = False,
    shuffle=True,
    random_state=None,
    test_size=0.2,
    verbose=True,
    augment=True,
    desired_shape: tuple[int, int] = (250, 250),
    augmentation_count: int = 10,
    augmentation_upto: int | None = None,
    augmentation_pipeline=None,
):
    X, Y = fetch_lfw_people(
        min_faces=min_faces,
        max_faces=max_faces,
        hard_limit=hard_limit,
        shuffle=shuffle,
        random_state=random_state,
        verbose=verbose,
    )

    tracker = tqdm.tqdm if verbose else identity

    __x = []
    for x in tracker(X):
        __x.append(x())
    # exit(1)

    __x = np.array(__x)

    # TODO: Normalize
    # __x = __x / 255

    x_train, x_test, y_train, y_test = train_test_split(
        __x,
        Y,
        test_size=test_size,
        random_state=random_state,
        stratify=Y,
    )

    if augment:
        if augmentation_upto is None:
            xy = []
            for i, x in tracker(enumerate(x_train)):
                aug_data = augment_data(
                    x,
                    desired_shape=desired_shape,
                    augmentation_count=augmentation_count,
                    augmentation_pipeline=augmentation_pipeline,
                )

                for data in aug_data:
                    data = Image.fromarray(data).convert("L")
                    data = np.array(data)
                    xy.append([data, y_train[i]])
                # print(xy)
                # fig, axes = plt.subplots(2, 5, figsize=(20, 10))
                # axes = axes.flatten()

                # for img, ax in zip(aug_data, axes):
                #     ax.imshow(img)
                #     ax.axis("off")
                # plt.tight_layout()
                # plt.show()

                # exit(1)
            xy_data = np.array(xy, dtype=object)
            x_train_data = xy_data[:, 0]
            x_train_data = x_train_data / 255
            y_train_data = xy_data[:, 1]
            x_train_data.dump("Dataset/x_train.npy")
            y_train_data.dump("Dataset/y_train.npy")
            x_test.dump("Dataset/x_test.npy")
            x_test = x_test / 255
            y_test.dump("Dataset/y_test.npy")
            # og_data = np.array(__x)
            # og_data.dump("Dataset/x_og.npy")
            # y_data.dump("Dataset/y.npy")
        else:
            xy = []
            idents = {}
            for i, y in enumerate(y_train):
                if y in idents:
                    idents[y].append(i)
                else:
                    idents[y] = [i]

            if augmentation_upto == 0:
                # augment upto max_faces
                if max_faces is None:
                    raise Exception("max_faces must be int if augmentation_upto is 0")

                for y in idents:
                    augmentation_len = max_faces - len(idents[y])
                    for _k in range(augmentation_len):
                        # choose random image
                        random_image = np.random.choice(idents[y])
                        # augment it
                        aug_data = augment_data(
                            x_train[random_image],
                            desired_shape=desired_shape,
                            augmentation_count=1,
                            augmentation_pipeline=augmentation_pipeline,
                        )
                        # add it to the dataset
                        xy.append([aug_data[0], y])
            elif augmentation_upto > 0:
                # augment upto augmentation_upto
                faces = len(idents[y])
                if augmentation_upto > faces:
                    for y in idents:
                        augmentation_len = augmentation_upto - len(idents[y])
                        for _k in range(augmentation_len):
                            # choose random image
                            random_image = np.random.choice(idents[y])
                            # augment it
                            aug_data = augment_data(
                                x_train[random_image],
                                desired_shape=desired_shape,
                                augmentation_count=1,
                                augmentation_pipeline=augmentation_pipeline,
                            )
                            # add it to the dataset
                            xy.append([aug_data[0], y])
            else:
                raise Exception("augmentation_upto must be bool or int")

            # add the original image with the label
            for i, x in tracker(enumerate(x_train)):
                cv_image = Image.fromarray(x).convert("L")
                cv_image = np.array(cv_image)
                xy.append([cv_image, y_train[i]])

            xy_data = np.array(xy, dtype=object)
            x_train_data = xy_data[:, 0]
            x_train_data = x_train_data / 255
            y_train_data = xy_data[:, 1]
            x_train_data.dump("Dataset/x_train.npy")
            y_train_data.dump("Dataset/y_train.npy")
            x_test.dump("Dataset/x_test.npy")
            x_test = x_test / 255
            y_test.dump("Dataset/y_test.npy")

    else:
        # x_data = np.array(__x)
        # x_data.dump("Dataset/x.npy")
        # Y.dump("Dataset/y.npy")
        x_train = x_train / 255
        x_train.dump("Dataset/x_train.npy")
        y_train.dump("Dataset/y_train.npy")
        x_test = x_test / 255
        x_test.dump("Dataset/x_test.npy")
        y_test.dump("Dataset/y_test.npy")


if __name__ == "__main__":
    export_dataset_objects(
        shuffle=False,
        min_faces=80,
        max_faces=100,
        hard_limit=False,
        augment=True,
        augmentation_upto=10,
    )
