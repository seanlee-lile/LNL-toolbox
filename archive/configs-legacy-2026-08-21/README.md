# Legacy configuration snapshot (2026-08-21)

This directory is an immutable recovery copy of all 92 YAML files that were under
`configs/` before the versioned configuration-contract migration.

- `manifest.json` records every relative path, byte count, and SHA-256 digest.
- The four subdirectories preserve the original layout exactly.
- These files are not discovered as built-in recipes and must not be edited in place.
- To recover one historical setting, copy it out and migrate it through the current
  configuration schema; do not run this archive as an active recipe directory.

