# Paired ionized calcium / pH

Fixed qualified support fails; no exposure contrasts computed. Actual execution: local.

[
  {
    "dataset": "eicu",
    "stage": "raw_calcium",
    "people": 150,
    "exposed": 55,
    "unexposed": 95,
    "hospitals": 30,
    "gate": true
  },
  {
    "dataset": "eicu",
    "stage": "qualified_joint_panel",
    "people": 7,
    "exposed": 4,
    "unexposed": 3,
    "hospitals": 3,
    "gate": false
  },
  {
    "dataset": "mimic",
    "stage": "raw_calcium",
    "people": 542,
    "exposed": 87,
    "unexposed": 455,
    "hospitals": null,
    "gate": true
  },
  {
    "dataset": "mimic",
    "stage": "qualified_joint_panel",
    "people": 526,
    "exposed": 86,
    "unexposed": 440,
    "hospitals": null,
    "gate": true
  }
]
