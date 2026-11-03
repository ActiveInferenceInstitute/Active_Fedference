# moving-observer

## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1.0

## ModelName
moving-observer

## ModelAnnotation
Native structured categorical specialization; no upstream backend equivalence claim.

## FedferenceStructured
{
  "schema": "fedference.structured/1",
  "model": {
    "kind": "moving-observer",
    "metadata": {
      "name": "moving-observer",
      "schema": "fedference.gnn/1",
      "precision": "float64-atol-1e-12-rtol-0",
      "capabilities": [
        "fedference.gnn/moving-observer/1"
      ],
      "schedule": [
        "predict",
        "observe"
      ]
    },
    "threat": {
      "states": [
        "left",
        "right"
      ],
      "observations": [
        "detected",
        "not_detected"
      ],
      "actions": [
        "stay",
        "left",
        "right"
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
          0.85,
          0.1,
          0.15,
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
          2,
          2,
          3
        ],
        "values": [
          1.0,
          1.0,
          1.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          1.0,
          1.0,
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
    "positions": [
      "0",
      "1",
      "2",
      "3"
    ],
    "position_transition": {
      "axes": [
        "next_position",
        "previous_position",
        "action"
      ],
      "shape": [
        4,
        4,
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
        1.0,
        1.0,
        0.0,
        1.0
      ]
    },
    "position_prior": {
      "axes": [
        "position"
      ],
      "shape": [
        4
      ],
      "values": [
        0.25,
        0.25,
        0.25,
        0.25
      ]
    },
    "observation_map": {
      "axes": [
        "observation",
        "threat",
        "position"
      ],
      "shape": [
        2,
        2,
        4
      ],
      "values": [
        0.85,
        0.85,
        0.1,
        0.1,
        0.1,
        0.1,
        0.85,
        0.85,
        0.15,
        0.15,
        0.9,
        0.9,
        0.9,
        0.9,
        0.15,
        0.15
      ]
    }
  },
  "native": {
    "capability": "native-moving-observer/1",
    "roles": {
      "threat": "latent-half",
      "position": "controlled-observer",
      "observation_map": "observation-given-threat-position"
    },
    "agent_positions": [
      0,
      2
    ],
    "fov_width": 2,
    "resize": "reflecting-line-half-sensor/1"
  }
}
