## pynq board setup

main nynq docs: https://pynq.readthedocs.io/en/latest/

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

fix Vitis HLS 2022.1 running on ubuntu 22.04

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

## How to create design

A+B hello world: https://www.youtube.com/watch?v=2ErFDGSv5EE

- create a custom IP in HLS in Vitis HLS (part name is xc7z020clg400-1)
- take code sample from the docs: https://pynq.readthedocs.io/en/latest/overlay_design_methodology/overlay_tutorial.html
- run synthesis to convert to HDL -> export RTL -> look at impl registers offsets xadd_hw.h
- create design -> add zynq PS -> run block automation
- tools -> settings -> change IP repo
- add custom ip (change name of the ip) -> run connection automation. -> save block design
- create wrapper for the block -> generate bitstream -> export block design to tcl (rename) and also get .hwh from .gen/hw_handoff and .bit from .runs
- move the 3 files to pynq/overlays/something_name

sample script:

```python
from pynq import Overlay
!ls /home/xilinx/pynq/overlays/

overlay = Overlay('/home/xilinx/pynq/overlays/adder/adder.bit')
help(overlay)

add_ip = overlay.scalar_add
help(add_ip)

add_ip.write(0x10, 4)
add_ip.write(0x18, 5)
add_ip.read(0x20)

add_ip.register_map
add_ip.register_map.a = 3
add_ip.register_map.b = 4
add_ip.register_map.c



# to wrap into a class
from pynq import DefaultIP

class AddDriver(DefaultIP):
    def __init__(self, description):
        super().__init__(description=description)

    bindto = ['xilinx.com:hls:add:1.0']

    def add(self, a, b):
        self.write(0x10, a)
        self.write(0x18, b)
        return self.read(0x20)

overlay = Overlay('/home/xilinx/pynq/overlays/adder/adder.bit')
help(overlay)

overlay.scalar_add.add(15,20)
```

Sample FIR filter: https://www.youtube.com/watch?v=PwG037LuNvA&

- set board to DDR
- create hierarchy

pynq changed code for array alloc:

```python
from pynq import allocate
input_buffer = allocate(shape=(n,), dtype='i4')
out_buffer = allocate(shape=(n,), dtype='i4')
input_buffer[:] = samples
n, samples
in_buffer = input_buffer

# Trigger the DMA transfer and wait for the result
import time
start_time = time.time()
dma.sendchannel.transfer(in_buffer)
dma.recvchannel.transfer(out_buffer)
dma.sendchannel.wait()
dma.recvchannel.wait()
stop_time = time.time()
hw_exec_time = stop_time-start_time
print('Hardware FIR execution time: ',hw_exec_time)
print('Hardware acceleration factor: ',sw_exec_time / hw_exec_time)

# Plot to the notebook
plot_to_notebook(t,samples,1000,out_signal=out_buffer)

# Free the buffers
in_buffer.close()
out_buffer.close()
```

Sigmoid function: https://github.com/fastmachinelearning/hls4ml/blob/main/hls4ml/templates/vivado/nnet_utils/nnet_activation.h
