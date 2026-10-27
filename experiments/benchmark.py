from time import perf_counter
import numpy as np
from Ex_1 import project_plane

# נפח של 500×500×500, עם ערכים בגודל בית אחד
M = np.full((500, 500, 500), 7, dtype=np.uint8)

a = [0, 0, 250]
a_second = [499, 0, 250]
b = [0, 499, 250]
c = [499, 499, 250]

# הרצת הכנה
project_plane(M, a, b, c, a_second)

repeats = 10

start = perf_counter()

for _ in range(repeats):
    E = project_plane(M, a, b, c, a_second)

elapsed = perf_counter() - start

print("Average seconds per frame:", elapsed / repeats)
print("Calculation FPS:", repeats / elapsed)