@dataclass
class ConstantInteraction:
    vx: float
    vd: float


@dataclass
class CoulombInteraction:
    pass


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
                    "Coulomb interaction takes no params"
                )