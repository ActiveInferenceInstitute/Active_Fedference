# GNN hybrid observation and target specification

## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1.0

## ModelName
hybrid-tracking-observation-model

## ModelAnnotation
Friston et al. (2024), federated belief sharing: bounded project-local hybrid
observation/target adapter, not reconstruction of the complete protocol.
The Gaussian mean is target position [m] and velocity observation bias [m/s];
covariance is sensor-noise covariance with entry units unit[i]*unit[j].
It is not the initial state prior or preference uncertainty. The native tracking
capability accepts zero velocity bias and diagonal covariance only. Symmetric
categorical A supplies cue accuracy. Identity B, zero C, and uniform D are fixed
native-protocol declarations; modifications are refused by tracking.
Native tracking retains its documented initial state and two-phase context
schedule, uses the declared complete configuration, and takes an explicit seed.
The separate Gaussian-conditioning operation treats these same numeric Gaussian
parameters as an arithmetic recovery witness; it is not a tracking result.
No generic PyMDP, continuous active-inference, efficacy, or scientific parity claim.

## FedferenceScientific
{
  "schema": "fedference.scientific/1",
  "model": {
    "kind": "hybrid-gaussian",
    "metadata": {
      "name": "hybrid-tracking-observation-model",
      "schema": "fedference.gnn/1",
      "precision": "float64-atol-1e-12-rtol-0",
      "capabilities": [
        "fedference.gnn/hybrid-gaussian/1"
      ],
      "schedule": [
        "predict",
        "observe"
      ]
    },
    "categorical": {
      "states": [
        "constant-velocity",
        "accelerating"
      ],
      "observations": [
        "constant-cue",
        "accelerating-cue"
      ],
      "actions": [
        "native-controller"
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
          0.15,
          0.15,
          0.85
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
          1,
          0,
          0,
          1
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
          0,
          0
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
    "coordinates": [
      "target_position",
      "velocity_observation_bias"
    ],
    "units": [
      "m",
      "m/s"
    ],
    "mean": {
      "axes": [
        "coordinate"
      ],
      "shape": [
        2
      ],
      "values": [
        1,
        0
      ]
    },
    "covariance": {
      "axes": [
        "coordinate",
        "coordinate_dual"
      ],
      "shape": [
        2,
        2
      ],
      "values": [
        0.04,
        0,
        0,
        0.01
      ]
    }
  },
  "native": {
    "capability": "native-hybrid-tracking/1",
    "field_roles": {
      "mean": "target-position-and-velocity-observation-bias",
      "covariance": "position-and-velocity-observation-noise"
    },
    "tracking": {
      "n_agents": 3,
      "horizon": 4,
      "process_var": 0.0025,
      "action_gain": 0.6,
      "max_acceleration": 0.4,
      "robustness": 0,
      "outlier_bias": 0
    }
  },
  "peers": []
}
