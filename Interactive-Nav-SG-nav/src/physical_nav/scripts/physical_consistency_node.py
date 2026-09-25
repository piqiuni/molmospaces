#!/usr/bin/env python3
"""Compatibility entry point for the physical deployment addon."""

from physical_nav_runtime import run

if __name__ == "__main__":
    run("physical_consistency_node.py")
