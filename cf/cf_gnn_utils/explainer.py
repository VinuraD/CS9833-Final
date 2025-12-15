from abc import ABC, abstractmethod
# from omegaconf import DictConfig
from torch_geometric.data import Data
import torch
import numpy as np
# from dataset import DataInfo

class Explainer(ABC):

    def __init__(self,datainfo) -> None:

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.verbose = False
        self.num_classes = None
        self.datainfo = datainfo

    @abstractmethod
    def explain(self, graph: Data, oracle, **kwargs)->dict:

        pass
    

    @abstractmethod
    def name(self):

        pass
    
    
    def set_reproducibility(self):

        # Reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
        np.random.seed(42)	