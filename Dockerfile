# ════════════════════════════════════════════════════════════════════════
# Contenedor c3-surface: 2DGS (2D Gaussian Splatting) → malla
# ════════════════════════════════════════════════════════════════════════
# Objetivo: tomar fotos + poses COLMAP y producir una MALLA por TSDF.
# Es el contenedor que el informe marca como "el más conflictivo con CUDA",
# por eso fijamos TODAS las versiones de forma EXACTA (nada de 'latest').
#
# Combinación PROBADA por la comunidad de 2DGS / 3DGS (evita el infierno CUDA):
#   - Base: CUDA 11.8 devel sobre Ubuntu 22.04 (trae nvcc para compilar)
#   - Python 3.10
#   - PyTorch 2.0.1 + cu118 (coincide EXACTO con la CUDA de la base)
#   - El rasterizador de 2DGS (diff-surfel-rasterization) se compila al FINAL
#
# Por qué CUDA 11.8 y no 12.x: 2DGS y su rasterizador surfel están más
# probados en 11.8; los issues de "CUDA mismatch" del repo de 3DGS casi
# siempre vienen de mezclar 12.x con PyTorch compilado para 11.8.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# TORCH_CUDA_ARCH_LIST: arquitecturas GPU que soportará el rasterizador.
#   8.6 = RTX 3090, 8.9 = RTX 4090, 8.0 = A100, 9.0 = H100.
#   Lo fijamos para que el binario compilado sirva en todas tus GPUs posibles.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda

# ── Paso 1: dependencias del sistema ──
#   git/wget/build-essential: para clonar y compilar
#   libgl1/libglib2.0: OpenCV y render headless
#   colmap NO se instala aquí (las poses llegan ya hechas desde R2)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates build-essential cmake ninja-build \
        libgl1 libglib2.0-0 libgomp1 \
        python3.10 python3.10-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# python3 → python3.10 y pip actualizado
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

# ── Paso 2: PyTorch 2.0.1 + cu118 (versión EXACTA, coincide con la base) ──
# Esta es la línea que más rompe si se deja libre. La fijamos.
RUN pip install --no-cache-dir \
        torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

# ── Paso 3: dependencias Python de 2DGS (versiones conocidas-buenas) ──
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

# ── Paso 5: compilar los submódulos CUDA UNO POR UNO (orden importa) ──
# Si alguno falla, el build PARA aquí con el error claro (mejor que seguir).
#   a) diff-surfel-rasterization: el rasterizador de surfels 2D (el núcleo)
RUN pip install --no-cache-dir ./submodules/diff-surfel-rasterization
#   b) simple-knn: vecinos cercanos para inicializar densidad
RUN pip install --no-cache-dir ./submodules/simple-knn

# ── Paso 6: verificación dentro de la imagen (que TODO importa bien) ──
# Si esto falla, la imagen no sirve y lo sabremos en el build, no en RunPod.
RUN python -c "import torch; assert torch.version.cuda=='11.8', torch.version.cuda; print('torch', torch.__version__, 'cuda', torch.version.cuda)" && \
    python -c "import diff_surfel_rasterization; print('diff-surfel-rasterization OK')" && \
    python -c "import simple_knn._C; print('simple-knn OK')" && \
    python -c "import open3d; print('open3d', open3d.__version__)" && \
    echo "=== imagen 2DGS lista (render-gs:v1) ==="

WORKDIR /workspace
CMD ["/bin/bash"]
