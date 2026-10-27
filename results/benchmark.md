# Benchmark results

Machine: Intel64 Family 6 Model 186 Stepping 3, GenuineIntel, Python 3.14.5, NumPy 2.5.4

| Implementation / plane | Volume | Output (rows x cols) | FPS |
|---|---|---|---:|
| Fast, oblique triangle | 128^3 uint8 | 143x129 | 2,422.3 |
| Fast, oblique triangle | 256^3 uint8 | 286x257 | 374.9 |
| Fast, oblique triangle | 500^3 uint8 | 559x501 | 83.5 |
| Fast, oblique triangle | 750^3 uint8 | 838x751 | 34.6 |
| Fast, oblique triangle | 500^3 uint16 | 559x501 | 103.2 |
| Fast, oblique triangle | 500^3 uint32 | 559x501 | 102.3 |
| Fast, axis-aligned square | 500^3 uint8 | 499x500 | 59.9 |
| Reference, oblique triangle | 128^3 uint8 | 143x129 | 24.5 |
