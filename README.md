for pynq board setup: https://pynq.readthedocs.io/en/latest/

docker engine setup on ubuntu (for finn): https://docs.docker.com/engine/install/ubuntu/

Then add `newgrp docker` to `.profile`

General board setup for package installation (note: dpu does not work on zynq-7000):
https://github.com/Xilinx/DPU-PYNQ

Xilinx Vitis 2022.1 works, 2023.1 doesn't...

```bash
sudo su
source /etc/profile.d/xrt_setup.sh
source /etc/profile.d/pynq_venv.sh
pip3 install pynq-dpu --no-build-isolation
```

**always append --no-build-isolation**
from finn-examples setup: https://github.com/Xilinx/finn-examples

finn setup: https://finn.readthedocs.io/en/latest/getting_started.html

if ssh-copy-id fails:

```bash
mkdir /tmp/home_dir/.ssh
```

to run in docker:

```bash
bash ./run_docker.sh
```

finn repo: https://github.com/xilinx/finn

bnn setup (): https://github.com/Xilinx/BNN-PYNQ/

custom pynq overlay (add a+b): https://www.youtube.com/watch?v=2ErFDGSv5EE

finn intro video: https://www.youtube.com/watch?v=zw2aG4PhzmA

## Data

https://github.com/numenta/NAB

## Baseline

Evaluate CPU implementation

1. [x] Decide on the models used (for fit-predict): Bolinger bands, Self-Attention, Linear regression
2. [x] Choose the dataset for evaluation (NAB as in the TranAD paper)
3. [x] Evaluate accuracy, latency and throughput

Evaluate GPU implementation

- Bolinger bands

- Linear regression on past observations

- Simple transformer (attention layer only)
