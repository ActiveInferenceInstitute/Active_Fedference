# GNN signed diagonal Gaussian site-factor specification

## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1.0

## ModelName
signed-diagonal-site-smoke

## ModelAnnotation
Mildner et al. (2025), FedGVI: project-local cavity, signed natural site, and
factor-replacement arithmetic. Precision increments have inverse unit products;
information increments have inverse coordinate units. Negative increments are
preserved; neither individual factor is a PMF or normalized Gaussian.
The native adapter consumes the primary and all peer SiteFactorModels, with
shared coordinate labels, units and protocol. This asset declares two nontrivial
clients. The global product and
cavity must both be proper Gaussians. One complete parallel round consumes every
explicit client posterior; the first cavity includes the nonzero peer site. This does not train a BNN or reproduce the source
client objective, dataset, or optimization schedule. Protocol parity remains
unresolved for client training. No generic PyMDP execution is supported.

## FedferenceScientific
{
  "schema": "fedference.scientific/1",
  "model": {
    "kind": "site-factor",
    "metadata": {
      "name": "signed-diagonal-site-smoke",
      "schema": "fedference.gnn/1",
      "precision": "float64-atol-1e-12-rtol-0",
      "capabilities": [
        "fedference.gnn/site-factor/1"
      ],
      "schedule": [
        "site-add"
      ]
    },
    "coordinates": [
      "weight_0",
      "weight_1"
    ],
    "units": [
      "1",
      "1"
    ],
    "protocol": "gaussian-natural-additive/1",
    "precision_increment": {
      "axes": [
        "coordinate",
        "coordinate_dual"
      ],
      "shape": [
        2,
        2
      ],
      "values": [
        -0.5,
        0,
        0,
        1
      ]
    },
    "information_increment": {
      "axes": [
        "coordinate"
      ],
      "shape": [
        2
      ],
      "values": [
        -0.25,
        0.5
      ]
    }
  },
  "native": {
    "capability": "native-diagonal-site/1",
    "representation": "signed-natural",
    "coordinates": [
      "weight_0",
      "weight_1"
    ],
    "units": [
      "1",
      "1"
    ],
    "prior_mean": [
      0,
      0
    ],
    "prior_covariance": [
      [
        0.5,
        0
      ],
      [
        0,
        0.25
      ]
    ],
    "schedule": "parallel",
    "client_means": [
      [
        0.5,
        -0.25
      ],
      [
        0.0,
        0.2
      ]
    ],
    "client_covariances": [
      [
        [
          0.25,
          0
        ],
        [
          0,
          0.5
        ]
      ],
      [
        [
          0.5,
          0.0
        ],
        [
          0.0,
          0.2
        ]
      ]
    ]
  },
  "peers": [
    {
      "kind": "site-factor",
      "metadata": {
        "name": "peer-diagonal-site",
        "schema": "fedference.gnn/1",
        "precision": "float64-atol-1e-12-rtol-0",
        "capabilities": [
          "fedference.gnn/site-factor/1"
        ],
        "schedule": [
          "site-add"
        ]
      },
      "coordinates": [
        "weight_0",
        "weight_1"
      ],
      "units": [
        "1",
        "1"
      ],
      "protocol": "gaussian-natural-additive/1",
      "precision_increment": {
        "axes": [
          "coordinate",
          "coordinate_dual"
        ],
        "shape": [
          2,
          2
        ],
        "values": [
          0.5,
          0.0,
          0.0,
          0.5
        ]
      },
      "information_increment": {
        "axes": [
          "coordinate"
        ],
        "shape": [
          2
        ],
        "values": [
          0.25,
          -0.25
        ]
      }
    }
  ]
}
