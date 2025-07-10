"""Backward compatibility wrapper for the Recursive Consciousness System."""

import asyncio
from rcs.main import main

if __name__ == "__main__":
    asyncio.run(main())
