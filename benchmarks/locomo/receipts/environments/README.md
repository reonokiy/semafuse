# External-worker Python inventories

These secret-free snapshots list the normalized installed distribution names
and versions seen by the actual invocation interpreters. The runner hashes the
same sorted maps with the domain `semafuse-python-installed-distributions/v1`
and binds that digest into each external method's run key.

The A-MEM environment contains 117 distributions. Its strict paper-backend
lock verifies 72 required exact pins, so `dependency_pins_verified=true` means
those required pins matched; it does not mean the environment was clean or had
no extra distributions. Mem0 and the reusable A-MEM package use non-strict
requirements files and therefore report `dependency_pins_verified=false`.

These inventories do not identify wheel bytes, indexes, platform tags, local
Git checkout commits, or PEP 610 provenance. They make the bound version map
inspectable, but the corresponding benchmark results remain development
evidence rather than hermetic or paper-reproduction runs.
