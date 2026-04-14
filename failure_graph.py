def cascade(temp, battery, signal):
    risks = []

    # NASA Standards: Battery below 30% is critical, Temp above 90C is critical
    if battery < 0.35:
        risks.append("BATTERY_LOW")
    if battery < 0.2:
        risks.append("POWER_CRITICAL")

    if temp > 0.7:
        risks.append("THERMAL_STRESS")
    if temp > 0.9:
        risks.append("THERMAL_OVERHEAT")

    if signal < 0.4:
        risks.append("SIGNAL_DEGRADATION")

    return risks