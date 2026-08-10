"""Allow ``python -m fedference_cli`` to invoke the installed CLI."""

from __future__ import annotations

from . import main

raise SystemExit(main())
