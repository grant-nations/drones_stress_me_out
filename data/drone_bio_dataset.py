import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from typing import Any, List, Union


class DroneBioDataset(Dataset):
    def __init__(self,
                 drone_bio_dfs: List[pd.DataFrame],
                 prev_stress_levels_dfs: List[pd.DataFrame],
                 demo_dfs: List[pd.DataFrame],
                 labels_dfs: Union[List[pd.DataFrame], None] = None) -> None:
        super().__init__()

        self.predict = labels_dfs is None

        self.drone_bio_dfs = drone_bio_dfs
        self.pre_stress_levels_dfs = prev_stress_levels_dfs
        self.demo_dfs = demo_dfs

        self.labels = None
        if not self.predict:
            self.labels_dfs = labels_dfs

    def __getitem__(self, index) -> Any:

        if self.predict:
            return (self.drone_bio_dfs[index].values.astype(np.float32),
                    self.pre_stress_levels_dfs[index].values.astype(np.float32),
                    self.demo_dfs[index].values.astype(np.float32))
        else:
            return (self.drone_bio_dfs[index].values.astype(np.float32),
                    self.pre_stress_levels_dfs[index].values.astype(np.float32),
                    self.demo_dfs[index].values.astype(np.float32),
                    self.labels_dfs[index].values.squeeze().astype(np.float32))

    def __len__(self) -> int:
        return len(self.drone_bio_dfs)
