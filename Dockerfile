# ════════════════════════════════════════════════════════════════════════
# Contenedor render-gs:v4 — MASt3R (poses feed-forward) + 2DGS → malla
# ════════════════════════════════════════════════════════════════════════
# CAMBIO MAYOR vs v3: las poses de cámara ya NO se calculan con COLMAP+SIFT
# (que fallaba en paredes blancas: solo 55/127 fotos, cuarto "fantasma" doble).
# Ahora las calcula MASt3R, un modelo de IA feed-forward que entiende la
# geometría de cada foto SIN depender de detectar "features" → registra casi
# todas las cámaras incluso en paredes lisas sin textura.
# El resto del pipeline (2DGS → malla por TSDF) NO cambia: MASt3R solo
# reemplaza el paso de poses.
# Versiones FIJAS (combinación CUDA probada): CUDA 11.8 + PyTorch 2.0.1 + cu118
# + Python 3.10. MASt3R corre sobre este mismo PyTorch (por eso NO usamos
# COLMAP 4.0, que exigiría CUDA 12).
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Arquitecturas GPU soportadas: 8.6=RTX3090, 8.9=RTX4090, 8.0=A100, 9.0=H100.
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV FORCE_CUDA=1
ENV CUDA_HOME=/usr/local/cuda

# ── Paso 1: dependencias del sistema + COLMAP ──
#   colmap (de apt): se mantiene SOLO por utilidades (p.ej. image_undistorter
#   si hiciera falta). Las poses ya NO las hace COLMAP, las hace MASt3R.
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
# Herramientas de build para compilar extensiones CUDA sin aislamiento.
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

# ── Paso 5: compilar submódulos CUDA de 2DGS (--no-build-isolation: usan el
# torch ya instalado; sin el flag fallan con "No module named 'torch'") ──
RUN pip install --no-cache-dir --no-build-isolation ./submodules/diff-surfel-rasterization
RUN pip install --no-cache-dir --no-build-isolation ./submodules/simple-knn

# ── Paso 6: MASt3R (motor de poses feed-forward) ──
# Se coloca al FINAL a propósito: si necesita ajustes, las capas pesadas de
# arriba (PyTorch, 2DGS) ya están en caché y no se recompilan.
# Clonamos con --recursive para traer los submódulos dust3r + croco.
WORKDIR /opt
RUN git clone --recursive https://github.com/naver/mast3r.git
WORKDIR /opt/mast3r

# Dependencias de runtime de MASt3R + DUSt3R (sobre el torch 2.0.1 ya instalado).
# NO instalamos gradio (es solo para la demo con interfaz; corremos headless).
# faiss-cpu + asmk = necesarios para el "retrieval" (decidir qué pares de fotos
# comparar entre las 127). cython lo necesita asmk para compilarse.
RUN pip install --no-cache-dir \
        roma \
        einops \
        "huggingface-hub[torch]>=0.22" \
        safetensors \
        matplotlib \
        scikit-learn \
        "pyglet<2" \
        tensorboard \
        cython \
        faiss-cpu

# Compilar la extensión CUDA 'curope' (acelera el cálculo de posiciones RoPE
# del transformer). Usa el torch instalado. Si fallara, MASt3R tiene un camino
# alternativo en PyTorch puro, pero lo construimos para velocidad.
RUN cd /opt/mast3r/dust3r/croco/models/curope && \
    python setup.py build_ext --inplace

# asmk (Aggregated Selective Match Kernels) para el retrieval por imagen.
RUN git clone https://github.com/jenicek/asmk.git /opt/asmk && \
    cd /opt/asmk/cython && cythonize *.pyx && \
    cd /opt/asmk && pip install --no-cache-dir -e .

# ── Paso 6b: HORNEAR los checkpoints de MASt3R (modelo de IA ya entrenado) ──
# Se descargan UNA vez aquí (en GitHub Actions, gratis) y quedan dentro de la
# imagen → NO se re-descargan en cada render en RunPod (ahorra tiempo y dinero).
#   - metric.pth  (~2.6GB): el modelo principal que estima geometría y poses.
#   - retrieval trainingfree.pth + codebook.pkl: para emparejar las fotos.
RUN mkdir -p /opt/mast3r/checkpoints && cd /opt/mast3r/checkpoints && \
    wget -q https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth && \
    wget -q https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth && \
    wget -q https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl && \
    ls -lh /opt/mast3r/checkpoints/

# MASt3R debe ser importable desde el worker (que corre en /workspace).
ENV PYTHONPATH=/opt/mast3r:/opt/mast3r/dust3r:/opt/2dgs

# ── Paso 7: verificación (torch SIEMPRE antes de las extensiones CUDA, para que
# carguen libc10.so de PyTorch) ──
RUN python -c "import torch; assert torch.version.cuda=='11.8', torch.version.cuda; print('torch', torch.__version__, 'cuda', torch.version.cuda); import diff_surfel_rasterization; print('diff-surfel-rasterization OK'); import simple_knn._C; print('simple-knn OK'); import open3d; print('open3d', open3d.__version__)" && \
    python -c "import sys; sys.path.insert(0,'/opt/mast3r'); sys.path.insert(0,'/opt/mast3r/dust3r'); from mast3r.model import AsymmetricMASt3R; print('MASt3R import OK'); import faiss; print('faiss OK'); import asmk; print('asmk OK')" && \
    test -f /opt/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth && echo "checkpoint metric OK" && \
    colmap -h > /dev/null 2>&1 && echo "colmap OK" && \
    echo "=== imagen MASt3R+2DGS lista (render-gs:v4) ==="

WORKDIR /workspace
CMD ["/bin/bash"]
