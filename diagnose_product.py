"""Print machine-readable collapse diagnostics for a TF-HDN FITS product."""

from __future__ import annotations

import argparse
import json

from kcor_ml.product_diagnostics import diagnose_product


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("product")
    parser.add_argument("--minimum-log-std", type=float, default=1.0e-3)
    parser.add_argument("--maximum-flat-radial-rms", type=float, default=2.0e-2)
    parser.add_argument("--maximum-noise-structure-correlation", type=float, default=0.5)
    arguments = parser.parse_args()
    result = diagnose_product(
        arguments.product,
        arguments.minimum_log_std,
        arguments.maximum_flat_radial_rms,
        arguments.maximum_noise_structure_correlation,
    )
    print(json.dumps(result, indent=2))
    if not result["eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
