FROM nvcr.io/nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# Copy files to /app/RFdiffusion in the container
COPY . /app/RFdiffusion/

# Install dependencies
RUN apt-get -q update \
  && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
  git \
  python3.9 \
  python3-pip \
  && python3.9 -m pip install -q -U --no-cache-dir pip \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get autoremove -y \
  && apt-get clean

# Install Python packages
RUN pip install -q --no-cache-dir \
  dgl==1.0.2+cu116 -f https://data.dgl.ai/wheels/cu116/repo.html \
  torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 \
  e3nn==0.3.3 \
  wandb==0.12.0 \
  pynvml==11.0.0 \
  git+https://github.com/NVIDIA/dllogger#egg=dllogger \
  decorator==5.1.0 \
  hydra-core==1.3.2 \
  pyrsistent==0.19.3 \
  numpy==1.26.4

# Install SE3Transformer and RFdiffusion packages in editable mode
RUN pip install -e /app/RFdiffusion/env/SE3Transformer --no-deps
RUN pip install -e /app/RFdiffusion --no-deps  

# Set the working directory to /root/RFdiffusion
WORKDIR /root/RFdiffusion

# Set environment variable
ENV DGLBACKEND="pytorch"

# Set entrypoint to run the inference script
ENTRYPOINT ["python3.9", "scripts/run_inference.py"]

