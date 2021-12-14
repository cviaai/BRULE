from abc import ABC, abstractmethod


class PathProvider(ABC):

    @staticmethod
    @abstractmethod
    def board() -> str: pass

    @staticmethod
    @abstractmethod
    def models() -> str: pass

    @staticmethod
    @abstractmethod
    def data() -> str: pass


class MyPath(PathProvider):

    @staticmethod
    def board() -> str:
        # path to tensorboard logdir
        pass

    @staticmethod
    def models() -> str:
        pass

    @staticmethod
    def data() -> str:
        pass


class Paths:

    default: PathProvider = MyPath







