from .core import BigSMILESbase


class _AbstractIterativeClass(BigSMILESbase):
    @property
    def generable(self):
        gen = True
        for child in self._children:
            gen = gen and child.generable
        return gen

    def generate_string(self, extension: bool) -> str:
        string = ""
        for child in self._children:
            string += child.generate_string(extension)
        return string


class BigSmiles(_AbstractIterativeClass):
    pass


class DotGeneration(_AbstractIterativeClass):
    pass


class DotSystemSize(BigSMILESbase):
    def __init__(self, children: list):
        super().__init__(children)

        self._molecular_weight: float = -1.0
        for child in self._children:
            if isinstance(child, float):
                if self._molecular_weight >= 0:
                    raise ValueError("Internal Error, please report on github.com")
                self._molecular_weight = child

    @property
    def molecular_weight(self) -> float:
        return self._molecular_weight

    def generate_string(self, extension: bool) -> str:
        string = ""
        if extension:
            string += f"|{self.molecular_weight}|"
        return string


class BigSmilesMolecule(_AbstractIterativeClass):
    pass
