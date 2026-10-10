from dataclasses import dataclass
from typing import Literal

@dataclass
class ConstantInteraction:
    vx: float
    vd: float


@dataclass
class CoulombInteraction:
    vx: float
    vd: float


@dataclass
class InteractionConfig:
    kind: Literal["constant", "coulomb"]

    params: object

    def validate(self):
        if self.kind == "constant":
            if not isinstance(
                self.params,
                ConstantInteraction
            ):
                raise ValueError(
                    "Constant interaction requires "
                    "ConstantInteraction params"
                )

        elif self.kind == "coulomb":
            if not isinstance(
                self.params,
                CoulombInteraction
            ):
                raise ValueError(
                    "Coulomb interaction requires "
                    "CoulombInteraction params"
                )