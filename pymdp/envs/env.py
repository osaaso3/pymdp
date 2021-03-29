from abc import ABC, abstractmethod


class Env(ABC):
    @abstractmethod
    def reset(self, obs=None):
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    def __str__(self):
        return "<{}>".format(type(self).__name__)