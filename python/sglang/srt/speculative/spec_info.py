from enum import IntEnum, auto


class SpeculativeAlgorithm(IntEnum):
    NONE = auto()
    NORMAL = auto()
    EAGLE = auto()
    EAGLE3 = auto()

    def is_none(self):
        return self == SpeculativeAlgorithm.NONE

    def is_eagle(self):
        return self == SpeculativeAlgorithm.EAGLE or self == SpeculativeAlgorithm.EAGLE3

    def is_eagle3(self):
        return self == SpeculativeAlgorithm.EAGLE3

    def is_not_eagle(self):
        return self == SpeculativeAlgorithm.NORMAL or self == SpeculativeAlgorithm.NONE

    def is_normal(self):
        return self == SpeculativeAlgorithm.NORMAL

    @staticmethod
    def from_string(name: str):
        name_map = {
            "EAGLE": SpeculativeAlgorithm.EAGLE,
            "EAGLE3": SpeculativeAlgorithm.EAGLE3,
            "NORMAL": SpeculativeAlgorithm.NORMAL,
            None: SpeculativeAlgorithm.NONE,
        }
        if name is not None:
            name = name.upper()
        return name_map[name]
