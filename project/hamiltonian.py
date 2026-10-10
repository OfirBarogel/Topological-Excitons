from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class Hamiltonian:
    kind: Literal["qwz", "haldane", "mos2"]
    params: object

    def validate(self):
        expected = {
            "qwz": QWZConfig,
            "haldane": HaldaneConfig,
            "mos2": MoS2Config,
        }

        expected_type = expected[self.kind]

        if not isinstance(self.params, expected_type):
            raise TypeError(
                f"Hamiltonian kind '{self.kind}' requires "
                f"{expected_type.__name__}, got "
                f"{type(self.params).__name__}"
            )

        self.params.validate()

@dataclass
class QWZConfig():
    M: float
    t: float = -1.0
    tt: float = 0.0
    def validate(self):
        pass

@dataclass
class HaldaneConfig():
    M: float
    phi: float
    t1: float = 1.0
    t2: float = 0.1
    
    def validate(self):
        pass

@dataclass
class SOCConfig:
    lambda_: float


@dataclass
class ZeemanConfig:
    B: tuple[float, float, float]


@dataclass
class ModifiedConfig:
    delta: float


@dataclass
class MoS2Config():
    terms: list[Literal["soc", "zeeman", "modified"]] = field(default_factory=list)

    soc: Optional[SOCConfig] = None
    zeeman: Optional[ZeemanConfig] = None
    modified: Optional[ModifiedConfig] = None

    def validate(self):

        if "soc" in self.terms and self.soc is None:
            raise ValueError(
                "SOC term enabled but soc parameters missing"
            )

        if "soc" not in self.terms and self.soc is not None:
            raise ValueError(
                "soc parameters provided but soc not enabled"
            )

        if "zeeman" in self.terms and self.zeeman is None:
            raise ValueError(
                "Zeeman term enabled but zeeman parameters missing"
            )

        if "zeeman" not in self.terms and self.zeeman is not None:
            raise ValueError(
                "zeeman parameters provided but zeeman not enabled"
            )

        if "modified" in self.terms and self.modified is None:
            raise ValueError(
                "Modified term enabled but modified parameters missing"
            )

        if "modified" not in self.terms and self.modified is not None:
            raise ValueError(
                "modified parameters provided but modified not enabled"
            )
