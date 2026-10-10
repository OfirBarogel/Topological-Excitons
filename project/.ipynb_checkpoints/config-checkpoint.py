from dataclasses import dataclass, asdict
from typing import Literal
import yaml

from .hamiltonian import Hamiltonian
from .interaction import InteractionConfig

@dataclass
class GridConfig:
    Handle: Literal["extend", "sew"] = "extend"
    Range: Literal["forward", "center"] = "forward"
    SeamTest: bool = False

@dataclass
class SimulationConfig:
    version: str

    hamiltonian: Hamiltonian
    interaction: InteractionConfig
    Grid: GridConfig
    
    U: float

    NQ: int = 32

    v_band: int = 0
    c_band: int = 1

    epsilon: float = 1e-14

    eps1: float = 0
    eps2: float = 0

    n_Fs: int = 1

    RunType: Literal["Local", "Cluster"] = "Local"

    def validate(self):
        self.hamiltonian.validate()
        self.interaction.validate()

        if self.NQ <= 0:
            raise ValueError("NQ must be positive")

        if self.c_band == self.v_band:
            raise ValueError(
                "c_band and v_band must differ"
            )
    
    def save(self, path):
        self.validate()

        with open(path, "w") as f:
            yaml.safe_dump(
                asdict(self),
                f,
                sort_keys=False
            )