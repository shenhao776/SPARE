import os
import numpy as np
from glob import glob
from os.path import join
import torch.utils.data as data
from sklearn.neighbors import NearestNeighbors

import datasets_util


class GeolocDataset(data.Dataset):
    def __init__(
        self,
        datasets_folder="datasets",
        dataset_name="pitts30k",
        split="train",
        positive_dist_threshold=25,
    ):

        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_folder = join(datasets_folder, dataset_name, "images", split)
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")

        gallery_folder = join(self.dataset_folder, "gallery")
        queries_folder = join(self.dataset_folder, "queries")
        if not os.path.exists(gallery_folder):
            raise FileNotFoundError(f"Folder {gallery_folder} does not exist")
        if not os.path.exists(queries_folder):
            raise FileNotFoundError(f"Folder {queries_folder} does not exist")
        self.gallery_paths = sorted(
            glob(join(gallery_folder, "**", "*.jpg"), recursive=True)
        )
        self.queries_paths = sorted(
            glob(join(queries_folder, "**", "*.jpg"), recursive=True)
        )

        self.gallery_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in self.gallery_paths]
        ).astype(float)
        self.queries_utms = np.array(
            [(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]
        ).astype(float)

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.gallery_utms)
        self.soft_positives_per_query = knn.radius_neighbors(
            self.queries_utms, radius=positive_dist_threshold, return_distance=False
        )
        self.images_paths = list(self.gallery_paths) + list(self.queries_paths)

        self.gallery_num = len(self.gallery_paths)
        self.queries_num = len(self.queries_paths)

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        img = datasets_util.open_image_and_apply_transform(image_path)
        return img, index

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #gallery: {self.gallery_num}; #queries: {self.queries_num} >"

    def get_positives(self):
        return self.soft_positives_per_query
