"""``fairprice-serve`` — launch the FastAPI pricing sidecar.

The sidecar exposes:

  GET  /health
  GET  /policy/version
  POST /price       (takes a LoanApplication, returns a recommended rate)
  GET  /audit/recent  (recent decisions for monitoring)

A Mifos / Fineract instance is expected to POST to ``/price`` when proposing
a rate for a floating-rate loan product.

Example::

    fairprice-serve --host 0.0.0.0 --port 8080
    fairprice-serve --policy rule_based            # default
    fairprice-serve --policy fair_cql --checkpoint runs/cql.pt
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from fairprice_mf.utils.api import build_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fairprice-serve", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--policy",
        default="rule_based",
        choices=["rule_based", "fair_cql"],
        help="Which policy to serve. fair_cql requires --checkpoint.",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    app = build_app(policy_name=args.policy, checkpoint=args.checkpoint)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
