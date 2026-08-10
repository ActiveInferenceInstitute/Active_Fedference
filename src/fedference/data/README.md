# Packaged compatibility data

`synthetic_tabular.csv` is the deterministic, synthetic compatibility fixture
used by `fedference.benchmark.run_tabular_benchmark()` when no path is supplied.
It is packaged into wheels and source distributions so the installed API does
not depend on a repository checkout.

This fixture is smoke/test data only. It is not an external-data replication or
manuscript evidence. Registered UCI runs use caller-owned, hash-verified caches
through `fedference.external_data`.
