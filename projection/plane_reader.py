"""Reading a list of planes from a CSV file."""
from __future__ import annotations

from .plane import Plane


def read_planes(csv_path: str) -> list[Plane]:
    """Parse a CSV where each row is one plane: 9 numbers (a,b,c) or
    12 numbers (a1,a2,b,c). Blank lines are skipped and a non-numeric first
    line is treated as a header. Errors report the offending line number.
    """
    planes = []
    with open(csv_path, "r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                values = [float(value) for value in line.split(",")]
            except ValueError:
                if line_number == 1:
                    continue
                raise ValueError(f"{csv_path}, line {line_number}: non-numeric value") from None
            try:
                planes.append(Plane.from_values(values))
            except ValueError as error:
                raise ValueError(f"{csv_path}, line {line_number}: {error}") from None

    if not planes:
        raise ValueError(f"{csv_path}: contains no planes")
    return planes
