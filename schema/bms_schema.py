from dataclasses import dataclass, fields
from typing import List

@dataclass
class BatteryRecord:
    CellVoltage: float
    CellTemperature: float
    InstantaneousCurrent: float
    AmbientTemperature: float
    CellSpecificGravity: float
    CellID: str = None
    Timestamp: str = None
    IsDead: int = None

def get_bms_columns() -> List[str]:
    return [f.name for f in fields(BatteryRecord)]