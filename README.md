# Growlight Calculator
This is a simple calculator to determine the number of grow lights needed for a given area. It takes into account the light output of the grow lights and the desired light intensity in micromoles per square meter per second ($μmol/m²/s$).
## Requirements
- Python 3.x
- PyTorch with CUDA support (if using GPU)
- NumPy
- Matplotlib
- Others (see `requirements.txt`)

## Installation
1. Clone the repository:
```bash
~$ git clone https://github.com/sqcode06/growlight-calculator.git
~$ cd growlight-calculator
```

2. Create a virtual environment (optional but recommended):
```bash
~$ python3 -m venv .venv
~$ source .venv/bin/activate
(.venv) ~$  # verify that you're on .venv now
```

3. Install the required packages:
```bash
(.venv) ~$ pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
(.venv) ~$ pip3 install requirements.txt
```
