# hierarchy

## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1.0

## ModelName
hierarchy

## ModelAnnotation
Native structured categorical specialization; no upstream backend equivalence claim.

## FedferenceStructured
{
  "schema": "fedference.structured/1",
  "model": {
    "kind": "hierarchical",
    "metadata": {
      "name": "hierarchy",
      "schema": "fedference.gnn/1",
      "precision": "float64-atol-1e-12-rtol-0",
      "capabilities": [
        "fedference.gnn/hierarchical/1"
      ],
      "schedule": [
        "predict",
        "observe"
      ]
    },
    "layers": [
      {
        "name": "regime",
        "categorical": {
          "states": [
            "stable",
            "volatile"
          ],
          "observations": [
            "stable",
            "volatile"
          ],
          "actions": [
            "stay"
          ],
          "A": {
            "axes": [
              "observation",
              "state"
            ],
            "shape": [
              2,
              2
            ],
            "values": [
              1.0,
              0.0,
              0.0,
              1.0
            ]
          },
          "B": {
            "axes": [
              "next_state",
              "previous_state",
              "action"
            ],
            "shape": [
              2,
              2,
              1
            ],
            "values": [
              1.0,
              0.0,
              0.0,
              1.0
            ]
          },
          "C": {
            "axes": [
              "observation"
            ],
            "shape": [
              2
            ],
            "values": [
              0.0,
              0.0
            ]
          },
          "D": {
            "axes": [
              "state"
            ],
            "shape": [
              2
            ],
            "values": [
              0.5,
              0.5
            ]
          }
        },
        "parent": null,
        "conditioned_prior": null
      },
      {
        "name": "meta-context",
        "categorical": {
          "states": [
            "low_threat",
            "high_threat"
          ],
          "observations": [
            "low_threat",
            "high_threat"
          ],
          "actions": [
            "stay"
          ],
          "A": {
            "axes": [
              "observation",
              "state"
            ],
            "shape": [
              2,
              2
            ],
            "values": [
              1.0,
              0.0,
              0.0,
              1.0
            ]
          },
          "B": {
            "axes": [
              "next_state",
              "previous_state",
              "action"
            ],
            "shape": [
              2,
              2,
              1
            ],
            "values": [
              1.0,
              0.0,
              0.0,
              1.0
            ]
          },
          "C": {
            "axes": [
              "observation"
            ],
            "shape": [
              2
            ],
            "values": [
              0.0,
              0.0
            ]
          },
          "D": {
            "axes": [
              "state"
            ],
            "shape": [
              2
            ],
            "values": [
              0.5,
              0.5
            ]
          }
        },
        "parent": "regime",
        "conditioned_prior": {
          "axes": [
            "child_state",
            "parent_state"
          ],
          "shape": [
            2,
            2
          ],
          "values": [
            1.0,
            0.0,
            0.0,
            1.0
          ]
        }
      },
      {
        "name": "context",
        "categorical": {
          "states": [
            "quiet",
            "alert"
          ],
          "observations": [
            "quiet",
            "alert"
          ],
          "actions": [
            "stay"
          ],
          "A": {
            "axes": [
              "observation",
              "state"
            ],
            "shape": [
              2,
              2
            ],
            "values": [
              1.0,
              0.0,
              0.0,
              1.0
            ]
          },
          "B": {
            "axes": [
              "next_state",
              "previous_state",
              "action"
            ],
            "shape": [
              2,
              2,
              1
            ],
            "values": [
              0.9,
              0.1,
              0.1,
              0.9
            ]
          },
          "C": {
            "axes": [
              "observation"
            ],
            "shape": [
              2
            ],
            "values": [
              0.0,
              0.0
            ]
          },
          "D": {
            "axes": [
              "state"
            ],
            "shape": [
              2
            ],
            "values": [
              0.5,
              0.5
            ]
          }
        },
        "parent": "meta-context",
        "conditioned_prior": {
          "axes": [
            "child_state",
            "parent_state"
          ],
          "shape": [
            2,
            2
          ],
          "values": [
            0.5,
            0.2,
            0.5,
            0.8
          ]
        }
      },
      {
        "name": "location",
        "categorical": {
          "states": [
            "cell_0",
            "cell_1",
            "cell_2",
            "cell_3",
            "cell_4",
            "cell_5",
            "cell_6",
            "cell_7",
            "cell_8"
          ],
          "observations": [
            "seen_0",
            "seen_1",
            "seen_2",
            "seen_3",
            "seen_4",
            "seen_5",
            "seen_6",
            "seen_7",
            "seen_8"
          ],
          "actions": [
            "still",
            "left",
            "right"
          ],
          "A": {
            "axes": [
              "observation",
              "state"
            ],
            "shape": [
              9,
              9
            ],
            "values": [
              0.9000000000000004,
              0.0125,
              0.0125,
              0.0125,
              0.0125,
              0.012499999999999999,
              0.012499999999999999,
              0.012499999999999997,
              0.012499999999999997,
              0.0125,
              0.9000000000000004,
              0.0125,
              0.0125,
              0.0125,
              0.012499999999999999,
              0.012499999999999999,
              0.012499999999999997,
              0.012499999999999997,
              0.0125,
              0.0125,
              0.9000000000000002,
              0.0125,
              0.0125,
              0.012499999999999999,
              0.012499999999999999,
              0.012499999999999997,
              0.012499999999999997,
              0.0125,
              0.0125,
              0.0125,
              0.9000000000000002,
              0.0125,
              0.012499999999999999,
              0.012499999999999999,
              0.012499999999999997,
              0.012499999999999997,
              0.0125,
              0.0125,
              0.0125,
              0.0125,
              0.9000000000000002,
              0.012499999999999999,
              0.012499999999999999,
              0.012499999999999997,
              0.012499999999999997,
              0.0125,
              0.0125,
              0.0125,
              0.0125,
              0.0125,
              0.9000000000000001,
              0.012499999999999999,
              0.012499999999999997,
              0.012499999999999997,
              0.0125,
              0.0125,
              0.0125,
              0.0125,
              0.0125,
              0.012499999999999999,
              0.9000000000000001,
              0.012499999999999997,
              0.012499999999999997,
              0.0125,
              0.0125,
              0.0125,
              0.0125,
              0.0125,
              0.012499999999999999,
              0.012499999999999999,
              0.9,
              0.012499999999999997,
              0.0125,
              0.0125,
              0.0125,
              0.0125,
              0.0125,
              0.012499999999999999,
              0.012499999999999999,
              0.012499999999999997,
              0.9
            ]
          },
          "B": {
            "axes": [
              "next_state",
              "previous_state",
              "action"
            ],
            "shape": [
              9,
              9,
              3
            ],
            "values": [
              1.0,
              1.0,
              0.0,
              0.0,
              1.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              1.0,
              1.0,
              0.0,
              0.0,
              0.0,
              1.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              1.0,
              1.0,
              0.0,
              1.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              1.0,
              1.0,
              0.0,
              0.0,
              1.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              1.0,
              1.0,
              0.0,
              0.0,
              0.0,
              1.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              1.0,
              1.0,
              0.0,
              1.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              1.0,
              1.0,
              0.0,
              0.0,
              1.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              1.0,
              1.0,
              0.0,
              0.0,
              0.0,
              1.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              1.0,
              1.0,
              0.0,
              1.0
            ]
          },
          "C": {
            "axes": [
              "observation"
            ],
            "shape": [
              9
            ],
            "values": [
              0.0,
              0.0,
              0.0,
              0.0,
              2.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          },
          "D": {
            "axes": [
              "state"
            ],
            "shape": [
              9
            ],
            "values": [
              0.0,
              0.0,
              0.0,
              0.0,
              1.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          }
        },
        "parent": "context",
        "conditioned_prior": {
          "axes": [
            "child_state",
            "parent_state"
          ],
          "shape": [
            9,
            2
          ],
          "values": [
            0.1111111111111111,
            0.05,
            0.1111111111111111,
            0.05,
            0.1111111111111111,
            0.05,
            0.1111111111111111,
            0.05,
            0.1111111111111111,
            0.6,
            0.1111111111111111,
            0.05,
            0.1111111111111111,
            0.05,
            0.1111111111111111,
            0.05,
            0.1111111111111111,
            0.05
          ]
        }
      }
    ]
  },
  "native": {
    "capability": "native-hierarchy/1",
    "messages": [
      "top-down-mixture",
      "bottom-up-posterior-evidence"
    ],
    "tasks": [
      {
        "task_id": "four-rooms",
        "grid": [
          ".........",
          "....#....",
          "....#....",
          ".........",
          "####.####",
          ".........",
          "....#....",
          "....#....",
          "........."
        ],
        "start": [
          1,
          1
        ],
        "goal": [
          7,
          7
        ],
        "horizon": 32,
        "key": null,
        "door": null
      },
      {
        "task_id": "key-door",
        "grid": [
          "#########",
          "#S..#..G#",
          "#...#...#",
          "#..K.D..#",
          "#...#...#",
          "#########"
        ],
        "start": [
          1,
          1
        ],
        "goal": [
          1,
          7
        ],
        "horizon": 28,
        "key": [
          3,
          3
        ],
        "door": [
          3,
          5
        ]
      }
    ]
  }
}
