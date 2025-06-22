from pydantic import BaseModel
from typing import Optional


class BMSSchema(BaseModel):
    """
    Pydantic schema for validating the structure and types of the BMS data.
    This ensures that data loaded into the pipeline conforms to expectations.
    """

    PacketID: int
    StartPacket: str
    DataIdentifier: str
    SiteID: str
    Time: str
    Date: str
    SerialNumber: str
    InstallationDate: str
    CellsConnectedCount: int
    ProblemCells: int
    CellNumber: int
    CellVoltage: float
    CellTemperature: float
    CellSpecificGravity: float
    CellServerTime: str
    StringVoltage: float
    SystemPeakCurrentInChargeOneCycle: float
    AverageDischargingCurrent: float
    AverageChargingCurrent: float
    AhInForOneChargeCycle: float
    AhOutForOneDischargeCycle: float
    CumulativeAHIn: float
    CumulativeAHOut: float
    ChargeTimeCycle: int
    DischargeTimeCycle: int
    TotalChargingEnergy: float
    TotalDischargingEnergy: float
    EveryHourAvgTemp: float
    CumulativeTotalAvgTempEveryHour: float
    ChargeOrDischargeCycle: int
    SocLatestValueForEveryCycle: float
    DodLatestValueForEveryCycle: float
    SystemPeakCurrentInDischargeOneCycle: float
    InstantaneousCurrent: float
    AmbientTemperature: float
    BatteryRunHours: int
    BMSBankDischargeCycle: bool
    BMSAmbientTemperatureHN: bool
    BMSSocLN: bool
    BMSStringVoltageLNH: int
    BMSStringCurrentHN: bool
    BMSBmsSedCommunication: bool
    BMSCellCommunication: bool
    BMSCellVoltageLN: bool
    BMSCellVoltageNH: bool
    BMSCellTemperatureHN: bool
    BMSBuzzer: bool
    ChargerID: int
    ChargerDeviceID: float
    ACVoltage: float
    ACCurrent: float
    Frequency: float
    Energy: bool
    ChargerInputMains: bool
    ChargerInputPhase: int
    ChargerDCVoltageOLN: int
    ChargerACVoltageULN: bool
    ChargerLoad: bool
    ChargerTrip: bool
    ChargerOutputMccb: bool
    ChargerBatteryCondition: bool
    ChargerTestPushButton: bool
    ChargerResetPushButton: bool
    ChargerAlarmSupplyFuse: bool
    ChargerFilterFuse: bool
    ChargerOutputFuse: bool
    ChargerInputFuse: bool
    ChargerRectifierFuse: str

    # Optional fields
    ProtocolVersion: Optional[float] = None
    PacketDateTime: Optional[str] = None
    DeviceID: Optional[str] = None
    BMSManufacturerID: Optional[int] = None
    ServerTime: Optional[float] = None
