FROM mambaorg/micromamba:1.4.9 as micromamba
FROM nvidia/cuda:11.0.3-devel

LABEL maintainer="Eduardo Arnold"

## Install libgl for Open3D
ENV DEBIAN_FRONTEND=noninteractive
RUN rm /etc/apt/sources.list.d/cuda.list && \
    rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get clean

## Setup micromamba
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"
ARG MAMBA_USER=mamba
ARG MAMBA_USER_ID=1000
ARG MAMBA_USER_GID=1000
ENV MAMBA_USER=$MAMBA_USER

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

## Install conda env
COPY environment.yml /opt/fastreg/
RUN micromamba install -y -n base -f /opt/fastreg/environment.yml && \
    micromamba clean --all --yes
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV ENV_NAME=base

## Copy remaining files and build pointnet2
COPY . /opt/fastreg/
ENV TORCH_CUDA_ARCH_LIST=Turing
RUN cd /opt/fastreg/lib/pointnet2/ && python setup.py install

WORKDIR /opt/fastreg/
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "/bin/bash"]