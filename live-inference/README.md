# Live Inference API - Battery Monitoring

## Required Input Fields

To perform live inference, submit a JSON with the following keys:

- CellVoltage (float)
- CellTemperature (float)
- InstantaneousCurrent (float)
- AmbientTemperature (float)
- CellSpecificGravity (float)
- CellID (str, optional)
- Timestamp (str, optional)
- IsDead (int, for failure prediction, optional)

## Output

Returns standardized JSON:
```json
{
  "status": "success" | "error",
  "message": "...",
  "data": { "prediction": [...] }
}
```