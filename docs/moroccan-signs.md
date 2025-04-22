# Moroccan Road Signs Reference

This document provides a comprehensive reference of the Moroccan road signs that our detection system can identify and classify.

## Sign Categories

Moroccan road signs are divided into three main categories:

1. **Regulatory Signs** - Indicate rules, regulations and prohibitions
2. **Warning Signs** - Alert drivers to potential hazards or changing conditions
3. **Information Signs** - Provide useful information for navigation and services

## Regulatory Signs

Regulatory signs in Morocco predominantly feature red circles and are mandatory to follow. These signs establish traffic rules and regulations.

### Speed Limit Signs

| Sign | Description | Detection Code |
|------|-------------|----------------|
| ![Speed Limit 30](../data/templates/regulatory/speed_limit_30.png) | Speed limit 30 km/h | R-SL-30 |
| ![Speed Limit 50](../data/templates/regulatory/speed_limit_50.png) | Speed limit 50 km/h | R-SL-50 |
| ![Speed Limit 60](../data/templates/regulatory/speed_limit_60.png) | Speed limit 60 km/h | R-SL-60 |
| ![Speed Limit 80](../data/templates/regulatory/speed_limit_80.png) | Speed limit 80 km/h | R-SL-80 |
| ![Speed Limit 100](../data/templates/regulatory/speed_limit_100.png) | Speed limit 100 km/h | R-SL-100 |
| ![Speed Limit 120](../data/templates/regulatory/speed_limit_120.png) | Speed limit 120 km/h | R-SL-120 |

### Prohibition Signs

| Sign | Description | Detection Code |
|------|-------------|----------------|
| ![No Entry](../data/templates/regulatory/no_entry.png) | No entry | R-NE-01 |
| ![No Parking](../data/templates/regulatory/no_parking.png) | No parking | R-NP-01 |
| ![No Stopping](../data/templates/regulatory/no_stopping.png) | No stopping | R-NS-01 |
| ![No Overtaking](../data/templates/regulatory/no_overtaking.png) | No overtaking | R-NO-01 |
| ![No Horn](../data/templates/regulatory/no_horn.png) | No horn | R-NH-01 |

### Mandatory Signs

| Sign | Description | Detection Code |
|------|-------------|----------------|
| ![Turn Right](../data/templates/regulatory/turn_right.png) | Turn right ahead | R-TR-01 |
| ![Turn Left](../data/templates/regulatory/turn_left.png) | Turn left ahead | R-TL-01 |
| ![Straight Ahead](../data/templates/regulatory/straight_ahead.png) | Straight ahead only | R-SA-01 |
| ![Roundabout](../data/templates/regulatory/roundabout.png) | Roundabout | R-RA-01 |

## Warning Signs

Warning signs in Morocco are typically triangular with red borders and feature a yellow or white background. These signs alert drivers to potential hazards ahead.

### Road Hazards

| Sign | Description | Detection Code |
|------|-------------|----------------|
| ![Dangerous Curve Right](../data/templates/warning/dangerous_curve_right.png) | Dangerous curve to the right | W-CR-01 |
| ![Dangerous Curve Left](../data/templates/warning/dangerous_curve_left.png) | Dangerous curve to the left | W-CL-01 |
| ![Double Curve](../data/templates/warning/double_curve.png) | Double curve | W-DC-01 |
| ![Slippery Road](../data/templates/warning/slippery_road.png) | Slippery road | W-SR-01 |
| ![Uneven Road](../data/templates/warning/uneven_road.png) | Uneven road | W-UR-01 |

### Road Users

| Sign | Description | Detection Code |
|------|-------------|----------------|
| ![Pedestrian Crossing](../data/templates/warning/pedestrian_crossing.png) | Pedestrian crossing | W-PC-01 |
| ![Children](../data/templates/warning/children.png) | Children crossing | W-CC-01 |
| ![Cyclists](../data/templates/warning/cyclists.png) | Cyclists crossing | W-CY-01 |
| ![Animals](../data/templates/warning/animals.png) | Wild animals | W-WA-01 |

### Road Conditions

| Sign | Description | Detection Code |
|------|-------------|----------------|
| ![Roadworks](../data/templates/warning/roadworks.png) | Road works | W-RW-01 |
| ![Traffic Signals](../data/templates/warning/traffic_signals.png) | Traffic signals ahead | W-TS-01 |
| ![Narrow Road](../data/templates/warning/narrow_road.png) | Road narrows | W-NR-01 |
| ![Falling Rocks](../data/templates/warning/falling_rocks.png) | Falling rocks | W-FR-01 |

## Information Signs

Information signs in Morocco typically feature blue or green backgrounds and provide guidance, directions, or information about facilities and services.

### Direction Signs

| Sign | Description | Detection Code |
|------|-------------|----------------|
| ![Motorway](../data/templates/information/motorway.png) | Motorway | I-MW-01 |
| ![Direction Sign](../data/templates/information/direction.png) | Direction sign | I-DS-01 |
| ![Exit](../data/templates/information/exit.png) | Exit | I-EX-01 |
| ![Distance](../data/templates/information/distance.png) | Distance information | I-DI-01 |

### Facility Signs

| Sign | Description | Detection Code |
|------|-------------|----------------|
| ![Parking](../data/templates/information/parking.png) | Parking | I-PA-01 |
| ![Hospital](../data/templates/information/hospital.png) | Hospital | I-HO-01 |
| ![Fuel Station](../data/templates/information/fuel.png) | Fuel station | I-FS-01 |
| ![Restaurant](../data/templates/information/restaurant.png) | Restaurant | I-RE-01 |
| ![Hotel](../data/templates/information/hotel.png) | Hotel | I-HT-01 |

## Sign Detection Performance

Our system performance varies by sign category. Below are the current detection accuracy metrics:

| Sign Category | Precision | Recall | F1 Score |
|---------------|-----------|--------|----------|
| Regulatory (Speed Limits) | 96.5% | 95.2% | 95.8% |
| Regulatory (Other) | 94.3% | 93.7% | 94.0% |
| Warning Signs | 92.8% | 91.5% | 92.1% |
| Information Signs | 90.6% | 89.3% | 89.9% |

## Custom Sign Templates

The system supports adding custom sign templates to improve detection accuracy or to add support for specialized or regional signs:

1. Prepare a clear, frontal image of the sign
2. Crop the image to include only the sign with minimal background
3. Use the Template Management feature in the desktop application
4. Add relevant metadata (category, description, detection code)
5. Save the template to the appropriate category folder

## Moroccan Road Sign Specificities

Moroccan road signs follow international standards with some regional differences:

- **Bilingual Text**: Many signs include text in both Arabic and French
- **Design Variations**: Some signs have minor design variations from European counterparts
- **Local Symbols**: Some signs include symbols specific to Moroccan culture or geography
- **Color Usage**: Standard international color schemes are used (red for prohibition, yellow for warning, etc.)

## References

1. Morocco's Highway Code (Code de la Route Marocain)
2. Moroccan Ministry of Equipment, Transport, Logistics and Water - Road Signage Guidelines
3. International Road Signs and Symbols (Vienna Convention)

## Contribution

We welcome contributions to improve our sign database. If you have high-quality images of Moroccan road signs that are not in our system, please consider contributing by:

1. Taking clear, frontal photographs of the signs
2. Submitting them through our [GitHub repository](https://github.com/dahbimoad/Moroccan-Road-Sign-Detection-System)
3. Including location data and sign category if possible

---

© 2025 Moad Dahbi | Moroccan Road Sign Detection System
Contact: dahbimoad1@gmail.com
