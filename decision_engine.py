def suggest_action(temp, battery, signal):
    actions = []

    if battery < 0.3:
        actions.append("LOW POWER MODE")

    if temp > 0.8:
        actions.append("THERMAL CONTROL")

    if signal < 0.3:
        actions.append("REALIGN ANTENNA")

    return actions if actions else ["NORMAL"]