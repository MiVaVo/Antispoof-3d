# Antifspoof-3d

Realtime (20+ FPS)  supervised CPU-only RGB and depth based approach for face spoofing detection (antispoofing). Face of a person is from a test set and was not contained in train set. Visual testing was done on IntelRealsense camera, but algorithm can work on any RGB+D cameras! This algorithm almost never gives False Positives. Bellow you can find error measurements, given different thresholds.

### Demo
[![Alt Text](Demo.gif)](https://www.youtube.com/watch?v=ek1j272iAmc)

### Run
```python
pip install requirements.txt
python main.py
```

### Validation results (for better threshold selection)
```
H0: face is normal(real)
th=0.832, FRR=0.074, FAR=0.001
th=0.855, FRR=0.075, FAR=0.000
th=0.879, FRR=0.082, FAR=0.000
th=0.903, FRR=0.092, FAR=0.000
th=0.926, FRR=0.108, FAR=0.000
th=0.950, FRR=0.123, FAR=0.000
```
