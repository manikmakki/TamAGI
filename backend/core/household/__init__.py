"""Opt-in ADR-0019 foundations. Standalone startup does not import this package.

The sandbox entry point uses newly provisioned fixtures only. Production enrollment,
runtime model transport, and household deployment remain separate qualification gates.
"""

from .store import HouseholdStore

__all__ = ["HouseholdStore"]
