from .preprocess import fetch_lfw_people
import tqdm
import numpy as np


def export_dataset_objects(
    min_faces: int | None = 20,
    max_faces: int | None = None,
    shuffle=True,
    random_state=None,
    verbose=True,
):
    X, Y = fetch_lfw_people(
        min_faces=min_faces,
        max_faces=max_faces,
        shuffle=shuffle,
        random_state=random_state,
        verbose=verbose,
    )

    __x = []
    for x in tqdm.tqdm(X):
        __x.append(x())

    x_data = np.array(__x)
    x_data.dump("Dataset/x.npy")
    Y.dump("Dataset/y.npy")


if __name__ == "__main__":
    export_dataset_objects(shuffle=False)
