"""Run a whole task suite, persist the results and compare runs."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
