from abc import ABC, abstractmethod

class BaseConverter(ABC):

    def __init__(
        self, 
        datapath: str
    ):
        self.datapath = datapath

    def convert(self):
        pass