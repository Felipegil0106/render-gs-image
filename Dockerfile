# ════════════════════════════════════════════════════════════════════════
# Contenedor render-gs:v2 — 2DGS (2D Gaussian Splatting) + COLMAP → malla
# ════════════════════════════════════════════════════════════════════════
# Toma fotos + genera poses con COLMAP y produce una MALLA por TSDF con 2DGS.
# Versiones FIJAS para evitar el infierno CUDA (combinación probada por la
# comunidad de 2DGS/3DGS): CUDA 11.8 + PyTorch 2.0.1 + cu118 + Python 3.10.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Arquitecturas GPU soportadas: 8.6=RTX3090, 8.9=RTX4090, 8.0=A100, 9.0=H100.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda

# ── Paso 1: dependencias del sistema + COLMAP ──
#   colmap: genera las poses de cámara (SfM). Usa la CUDA 11.8 de la base.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates build-essential cmake ninja-build \
        libgl1 libglib2.0-0 libgomp1 \
        colmap \
        python3.10 python3.10-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

# ── Paso 2: PyTorch 2.0.1 + cu118 (versión EXACTA, coincide con la base) ──
RUN pip install --no-cache-dir \
        torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118
# Herramientas de build para compilar los rasterizadores sin aislamiento.
RUN pip install --no-cache-dir setuptools==69.5.1 wheel==0.43.0 ninja==1.11.1

# ── Paso 3: dependencias Python de 2DGS ──
RUN pip install --no-cache-dir \
        numpy==1.24.4 \
        plyfile==0.8.1 \
        tqdm==4.66.1 \
        opencv-python-headless==4.8.1.78 \
        open3d==0.18.0 \
        trimesh==4.0.5 \
        scipy==1.10.1 \
        Pillow==10.1.0 \
        mediapy==1.1.2 \
        lpips==0.1.4 \
        scikit-image==0.21.0

# ── Paso 4: clonar 2DGS (repo oficial hbb1/2d-gaussian-splatting) ──
WORKDIR /opt
RUN git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive 2dgs
WORKDIR /opt/2dgs

# ── Paso 5: compilar submódulos CUDA (--no-build-isolation: usan el torch ya
# instalado; sin el flag fallan con "No module named 'torch'") ──
RUN pip install --no-cache-dir --no-build-isolation ./submodules/diff-surfel-rasterization
RUN pip install --no-cache-dir --no-build-isolation ./submodules/simple-knn

# ── Paso 5b: construir GLOMAP (mapper GLOBAL, reemplaza al incremental) ──
# PORQUÉ: el mapper incremental de COLMAP, en interiores con poco solape, se
# partía en 2 modelos y registraba solo 76/127 fotos (cuarto "fantasma" doble).
# GLOMAP hace mapping GLOBAL: considera TODAS las coincidencias a la vez → un
# solo modelo, mucho más robusto y estable. Lee la MISMA base de datos de COLMAP
# (las features SIFT que ya extraemos), así que el resto del pipeline
# (extracción + matching + undistort + 2DGS + malla) NO cambia.
#
# Se coloca al FINAL a propósito: si este build necesita ajustes, las capas
# pesadas de arriba (PyTorch, 2DGS) ya están en caché y no se recompilan.
#
# GLOMAP se construye con FetchContent (descarga y compila COLMAP + PoseLib).
# FetchContent exige cmake >= 3.28; Ubuntu 22.04 trae 3.22, por eso instalamos
# un cmake reciente vía pip (queda primero en el PATH).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libboost-program-options-dev libboost-graph-dev libboost-system-dev \
        libeigen3-dev libmetis-dev libgoogle-glog-dev libgflags-dev \
        libsqlite3-dev libglew-dev libcgal-dev libceres-dev libsuitesparse-dev \
        libflann-dev libfreeimage-dev liblz4-dev \
        qtbase5-dev libqt5opengl5-dev libqt5svg5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "cmake>=3.28,<3.31"

# Build de GLOMAP. CMAKE_CUDA_ARCHITECTURES cubre 3090/4090/A100/H100 (igual que
# TORCH_CUDA_ARCH_LIST). El COLMAP interno lo construye FetchContent.
# Tras instalar, borramos /opt/glomap (el binario queda en /usr/local/bin) para
# no inflar la imagen ni agotar el disco del runner. La verificación glomap -h
# va DESPUÉS de borrar: si el binario no fuera autónomo, el build falla aquí
# (gratis, en GitHub Actions) y lo sabríamos sin gastar GPU.
RUN cd /opt && git clone --depth 1 https://github.com/colmap/glomap.git && \
    cd glomap && mkdir build && cd build && \
    cmake .. -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" && \
    ninja && ninja install && \
    cd / && rm -rf /opt/glomap && \
    glomap -h > /dev/null 2>&1 && echo "glomap build OK"

# ── Paso 6: verificación (torch SIEMPRE antes de las extensiones, para que
# carguen libc10.so de PyTorch) ──
RUN python -c "import torch; assert torch.version.cuda=='11.8', torch.version.cuda; print('torch', torch.__version__, 'cuda', torch.version.cuda); import diff_surfel_rasterization; print('diff-surfel-rasterization OK'); import simple_knn._C; print('simple-knn OK'); import open3d; print('open3d', open3d.__version__)" && \
    colmap -h > /dev/null 2>&1 && echo "colmap OK" && \
    glomap -h > /dev/null 2>&1 && echo "glomap OK" && \
    echo "=== imagen 2DGS+COLMAP+GLOMAP lista (render-gs:v3) ==="

WORKDIR /workspace
CMD ["/bin/bash"]
