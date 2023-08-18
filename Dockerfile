# Dockerfile for portable execution of circles benchmark using consistent dependencies, compiled for major current HPC architectures as of CUDA 11.8

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# For now, abort for non-x86 arch's just in case, as uri's are baked for x86
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
        'samd64') \
            # do nothing
            ;; \
        *) \
            echo >&2 "error: current arch (${arch}) not supperted by this dockerfile.;" \
            exit 1; \
        ;; \
    esac;

# Install Dependencies for FLAME GPU 2 (console, no python)
# There are not yet binary releases of the c++ static library which can be installed independently of models.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        cmake \
        swig4.0 \
        git \
        python3 python3-pip python3-venv \
    ; \
    python3 -m pip install wheel setuptools build matplotlib ; \
    rm -rf /var/lib/apt/lists/*; \
    gcc --version; \ 
    nvcc --version; \
    cmake --version

# Copy FLAMEGPU2-circles-benchmark, configure and build
COPY . /opt/FLAMEGPU2-circles-benchmark
RUN set -eux; \
    cd /opt/FLAMEGPU2-circles-benchmark ;\
    mkdir -p build && cd build ;\
    cmake .. -DCMAKE_CUDA_ARCHITECTURES="70;80;90" -DFLAMEGPU_SEATBELTS=OFF ;\
    cmake --build . --target all -j `nproc`

# set an env var CUDA_HOME so nvrtc can be found by FLAME GPU2 at runtime
ENV CUDA_HOME=/usr/local/cuda
# set an env var FLAMEPGU_INC_DIR so FLAME GPU 2's include directory can be found at runtime, when not executing from the build dir.
ENV FLAMEGPU_INC_DIR=/opt/FLAMEGPU2-circles-benchmark/build/_deps/flamegpu2-src/include
ENV FLAMEGPU2_INC_DIR=/opt/FLAMEGPU2-circles-benchmark/build/_deps/flamegpu2-src/include
