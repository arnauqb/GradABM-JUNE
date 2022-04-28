from abc import ABC
import re

from torch_june.utils import read_date


class Policy(ABC):
    def __init__(self, start_date, end_date):
        self.spec = self.get_spec()
        self.start_date = read_date(start_date)
        self.end_date = read_date(end_date)

    def apply(self):
        raise NotImplementedError

    def get_spec(self) -> str:
        """
        Returns the speciailization of the policy.
        """
        return re.sub(r"(?<!^)(?=[A-Z])", "_", self.__class__.__name__).lower()
