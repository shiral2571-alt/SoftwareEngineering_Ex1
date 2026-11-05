"""Reading a list of planes from a CSV file.

The format is the lecturer's example file (one plane per row):

    Number of Points (3 or 4),x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4
    3,ax,ay,az,bx,by,bz,cx,cy,cz,,,
    4,a1x,a1y,a1z,bx,by,bz,a2x,a2y,a2z,cx,cy,cz

so a triangle is (a, b, c) and a trapezoid is (a1, b, a2, c): the two legs
are point 1 -> point 2 and point 3 -> point 4. Lines starting with '#' are
comments. Rows without the leading count
(9 numbers a,b,c or 12 numbers a1,a2,b,c, the format of the first version)
are still accepted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .plane import Plane

DELIMITERS = re.compile(r"[,;\t]")   # comma, semicolon (Excel) or tab


@dataclass(frozen=True)
class PlaneRow:
    """One data row of the CSV: either a valid plane or the reason it is not."""
    line_number: int
    plane: Plane | None
    error: str | None = None


def read_plane_rows(csv_path: str) -> list[PlaneRow]:
    """Read every data row, keeping invalid rows (with their error) in order.

    Problems with the file itself (missing, unreadable, no rows) raise.
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue                # blank line or comment
            if line_number == 1 and any(ch.isalpha() for ch in line):
                continue                # header row, e.g. "Number of Points (3 or 4),x1,..."
            try:
                rows.append(PlaneRow(line_number, _parse_plane(line)))
            except ValueError as error:
                rows.append(PlaneRow(line_number, None, str(error)))

    if not rows:
        raise ValueError(f"{csv_path}: contains no planes")
    return rows


def read_planes(csv_path: str) -> list[Plane]:
    """Strict version: every row must be a valid plane, otherwise raise
    ValueError with the line number and the reason."""
    planes = []
    for row in read_plane_rows(csv_path):
        if row.plane is None:
            raise ValueError(f"{csv_path}, line {row.line_number}: {row.error}")
        planes.append(row.plane)
    return planes


def _parse_plane(line: str) -> Plane:
    fields = DELIMITERS.split(line) if DELIMITERS.search(line) else line.split()
    fields = [field.strip() for field in fields]
    while fields and not fields[-1]:
        fields.pop()                    # empty x4,y4,z4 columns of a triangle
    try:
        values = [float(field) for field in fields]
    except ValueError:
        raise ValueError("non-numeric or empty value") from None

    count = len(values)
    if count in (10, 13) and values[0] == (count - 1) // 3:
        points = [values[i:i + 3] for i in range(1, count, 3)]
        if count == 10:
            return Plane.triangle(*points)
        a1, b, a2, c = points
        return Plane.trapezoid(a1, a2, b, c)
    if count in (9, 12):
        return Plane.from_values(values)
    raise ValueError("expected '3' and 9 coordinates or '4' and 12 coordinates, "
                     f"got {count} values")
