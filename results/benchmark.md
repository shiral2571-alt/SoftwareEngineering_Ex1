# Benchmark results

Machine: Intel64 Family 6 Model 186 Stepping 3, GenuineIntel, Python 3.14.5, NumPy 2.5.4

| Implementation / plane | Volume | Output (rows x cols) | FPS |
|---|---|---|---:|
| Fast, oblique triangle | 128^3 uint8 | 143x129 | 1,364.6 |
| Fast, oblique triangle | 256^3 uint8 | 286x257 | 457.2 |
| Fast, oblique triangle | 500^3 uint8 | 559x501 | 173.3 |
| Fast, oblique triangle | 750^3 uint8 | 838x751 | 88.0 |
| Fast, oblique triangle | 500^3 uint16 | 559x501 | 188.0 |
| Fast, oblique triangle | 500^3 uint32 | 559x501 | 176.1 |
| Fast, axis-aligned square | 500^3 uint8 | 499x500 | 163.8 |
| Fast 1 thread, oblique triangle | 500^3 uint8 | 559x501 | 142.6 |
| Fast 1 thread, axis-aligned square | 500^3 uint8 | 499x500 | 88.0 |
| Reference, oblique triangle | 128^3 uint8 | 143x129 | 24.0 |
