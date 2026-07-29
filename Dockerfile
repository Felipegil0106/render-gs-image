# ════════════════════════════════════════════════════════════════════════
# Contenedor gaussian-mesh:v4.2 — MASt3R + 2DGS + OpenMVS + BLOQUE B
# FIX v4.1: numpy queda en 1.26.4 en TODA la imagen (v4 lo bajaba a 1.24.4
# y rompía faiss, que exige >=1.25; scipy exige <1.27 → 1.26.4 es el punto
# exacto, y es el que la v3 usa en producción). faiss y asmk quedan pineados.
# FIX v4.2: la imagen escribe su versión en /opt/IMAGE_TAG y el worker la
# imprime en el banner del log → nunca más adivinar QUÉ imagen corrió el pod.
#   (priors monoculares de profundidad/normales + bundle adjustment de poses)
# ════════════════════════════════════════════════════════════════════════
# NUEVO EN v4 (Bloque B):
#   · pycolmap 4.1.0  → PASO 2b del worker: pulir poses MASt3R (nitidez textura)
#   · Depth Anything V2 metric-indoor vitb (~390MB, horneado) → prior profundidad
#   · DSINE ddedde1 (~291MB, horneado; opcional) → prior de normales
#   · repos de ambos modelos PINEADOS por SHA + xatlas/geffnet horneados
#   · 2DGS pineado al commit validado (el parche de priors depende de sus anclas)
# Todo lo de v3 queda IGUAL (mismas capas, mismo orden → caché aprovechable).
# CAMBIO MAYOR vs v2: las poses de cámara ya NO se calculan con COLMAP+SIFT
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
        numpy==1.26.4 \
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
RUN git clone --recursive https://github.com/hbb1/2d-gaussian-splatting.git 2dgs && \
    cd 2dgs && git checkout 335ad612f2e783a4e57b9cbc4d1e167bd599fc98 && \
    git submodule update --init --recursive
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
#
# CRÍTICO: usamos un archivo de "constraints" que CLAVA torch/torchvision en su
# versión exacta, para que NINGUNA de estas librerías intente ACTUALIZAR PyTorch.
# (En el primer build, "huggingface-hub[torch]" arrastró un torch de CUDA 13.0
# que rompía todo: las extensiones CUDA de 2DGS y curope se compilan contra el
# torch de CUDA 11.8, y un torch 13.0 las invalida.) Por eso también quitamos el
# extra [torch] de huggingface-hub. El assert final aborta el build (gratis) si
# algo cambió torch.
RUN printf 'torch==2.0.1\ntorchvision==0.15.2\nnumpy==1.26.4\n' > /opt/torch-constraints.txt && \
    pip install --no-cache-dir -c /opt/torch-constraints.txt \
        roma \
        einops \
        "huggingface-hub>=0.22" \
        safetensors \
        matplotlib \
        scikit-learn \
        "pyglet<2" \
        tensorboard \
        cython \
        faiss-cpu==1.14.3 && \
    python -c "import torch; assert torch.version.cuda=='11.8', 'torch fue cambiado a CUDA '+str(torch.version.cuda); print('OK: torch sigue en 2.0.1 / CUDA 11.8')"

# Compilar la extensión CUDA 'curope' (acelera el cálculo de posiciones RoPE
# del transformer). Usa el torch instalado. Si fallara, MASt3R tiene un camino
# alternativo en PyTorch puro, pero lo construimos para velocidad.
RUN cd /opt/mast3r/dust3r/croco/models/curope && \
    python setup.py build_ext --inplace

# asmk (Aggregated Selective Match Kernels) para el retrieval por imagen.
RUN git clone https://github.com/jenicek/asmk.git /opt/asmk && \
    cd /opt/asmk && git checkout 2a96d9c03a841dffdfddabc699a20512dcd09363 && \
    cd /opt/asmk/cython && cythonize *.pyx && \
    cd /opt/asmk && pip install --no-cache-dir -c /opt/torch-constraints.txt -e .

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

# ── Paso 6b: OpenMVS (TextureMesh) — HORNEAR TEXTURA REAL desde las fotos ──
# Recuperamos la herramienta de texturizado PROBADA del pipeline viejo. La malla
# de 2DGS trae solo COLOR POR VÉRTICE (baja frecuencia) → objetos "plásticos".
# OpenMVS TextureMesh proyecta las fotos sobre la malla y genera una IMAGEN de
# textura de alta resolución (UV atlas) → objetos con su textura REAL.
# Build SIN CUDA (TextureMesh es CPU). Receta basada en el buildInDocker.sh
# oficial de cdcseacave/openMVS. Binarios quedan en /usr/local/bin/OpenMVS/.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
        libpng-dev libjpeg-dev libtiff-dev \
        libglu1-mesa-dev libglew-dev libglfw3-dev \
        libboost-iostreams-dev libboost-program-options-dev \
        libboost-system-dev libboost-serialization-dev libboost-thread-dev \
        libopencv-dev libgmp-dev libmpfr-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Eigen 3.4 (solo cabeceras) desde fuente — igual que el build oficial
RUN cd /tmp && git clone --depth 1 https://gitlab.com/libeigen/eigen --branch 3.4 && \
    mkdir eigen_build && cd eigen_build && \
    cmake ../eigen >/dev/null 2>&1 && make >/dev/null 2>&1 && make install >/dev/null 2>&1 && \
    cd /tmp && rm -rf eigen_build eigen

# CGAL 6.0.1 (solo cabeceras) desde fuente — igual que el build oficial
RUN cd /tmp && git clone --depth 1 https://github.com/cgal/cgal --branch v6.0.1 && \
    mkdir cgal_build && cd cgal_build && \
    cmake ../cgal >/dev/null 2>&1 && make >/dev/null 2>&1 && make install >/dev/null 2>&1 && \
    cd /tmp && rm -rf cgal_build cgal

# nanoflann (solo cabeceras) — OpenMVS lo requiere (FIND_PACKAGE nanoflann REQUIRED).
# Esta era la pieza que faltaba: el build llegaba al paso de OpenMVS y fallaba aquí.
# Lo instalamos desde fuente para garantizar que quede el nanoflannConfig.cmake.
RUN cd /tmp && git clone --depth 1 https://github.com/jlblancoc/nanoflann.git && \
    mkdir nanoflann_build && cd nanoflann_build && \
    cmake ../nanoflann -DNANOFLANN_BUILD_EXAMPLES=OFF -DNANOFLANN_BUILD_TESTS=OFF >/dev/null 2>&1 && \
    make install >/dev/null 2>&1 && \
    cd /tmp && rm -rf nanoflann_build nanoflann

# VCGLib (cabeceras) + OpenMVS (rama master estable, SIN CUDA)
# DOS PARCHES a bugs de OpenMVS master con OpenCV 4.5.4 (la de Ubuntu 22.04):
#  1) libjxl marcado REQUIRED en su CMake aunque no lo necesitamos (fotos JPG/PNG):
#     quitamos el REQUIRED del pkg_check_modules → JPEG XL queda opcional.
#  2) Types.inl usa cv::IMWRITE_JPEGXL_QUALITY (solo existe en OpenCV 4.7+) dentro
#     de un bloque .jxl que NUNCA usamos. Lo cambiamos por cv::IMWRITE_JPEG_QUALITY
#     (que sí existe en 4.5.4) → compila; ese bloque es código muerto para nosotros.
RUN cd /opt && git clone --depth 1 https://github.com/cdcseacave/VCG.git vcglib && \
    git clone --depth 1 https://github.com/cdcseacave/openMVS.git --branch master && \
    sed -i 's/ REQUIRED IMPORTED_TARGET/ IMPORTED_TARGET/' openMVS/libs/IO/CMakeLists.txt && \
    sed -i 's/cv::IMWRITE_JPEGXL_QUALITY/cv::IMWRITE_JPEG_QUALITY/' openMVS/libs/Common/Types.inl && \
    mkdir openMVS_build && cd openMVS_build && \
    cmake ../openMVS -DCMAKE_BUILD_TYPE=Release -DVCG_ROOT=/opt/vcglib -DOpenMVS_USE_CUDA=OFF && \
    make -j"$(nproc)" && make install && \
    cd /opt && rm -rf openMVS_build vcglib

# Los binarios de OpenMVS (InterfaceCOLMAP, TextureMesh, …) al PATH
ENV PATH=/usr/local/bin/OpenMVS:$PATH

# ── Paso 7: verificación (torch SIEMPRE antes de las extensiones CUDA, para que
# carguen libc10.so de PyTorch) ──
RUN python -c "import torch; assert torch.version.cuda=='11.8', torch.version.cuda; print('torch', torch.__version__, 'cuda', torch.version.cuda); import diff_surfel_rasterization; print('diff-surfel-rasterization OK'); import simple_knn._C; print('simple-knn OK'); import open3d; print('open3d', open3d.__version__)" && \
    python -c "import sys; sys.path.insert(0,'/opt/mast3r'); sys.path.insert(0,'/opt/mast3r/dust3r'); from mast3r.model import AsymmetricMASt3R; print('MASt3R import OK'); import faiss; print('faiss OK'); import asmk; print('asmk OK')" && \
    test -f /opt/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth && echo "checkpoint metric OK" && \
    colmap -h > /dev/null 2>&1 && echo "colmap OK" && \
    test -x /usr/local/bin/OpenMVS/InterfaceCOLMAP && echo "OpenMVS InterfaceCOLMAP OK" && \
    test -x /usr/local/bin/OpenMVS/TextureMesh && echo "OpenMVS TextureMesh OK" && \
    echo "=== imagen MASt3R+2DGS+OpenMVS lista ==="


# ── Paso 8: BLOQUE B — priors monoculares + refinamiento de poses (v4) ──
# (a) pycolmap: bundle adjustment clásico para PULIR las poses de MASt3R
#     (quita el temblor de ~0.1° que emborrona la textura al promediar vistas).
# (b) Depth Anything V2 (métrico, interiores) + DSINE: por cada foto estiman
#     profundidad en metros y la orientación de cada pixel. El worker se los
#     pasa al entrenamiento de 2DGS para CERRAR huecos y dejar el techo y las
#     paredes lisas continuos (donde las fotos solas no anclan geometría).
# (c) xatlas (antes se instalaba en cada render; ahora queda horneado) y
#     geffnet (backbone de DSINE).
# El archivo de constraints CLAVA torch/torchvision/numpy para que NADA de
# esto actualice PyTorch (lección aprendida del primer build: un torch de
# CUDA 13 invalida las extensiones CUDA compiladas). El assert aborta gratis.
RUN printf 'torch==2.0.1\ntorchvision==0.15.2\nnumpy==1.26.4\n' > /opt/pip-constraints-v4.txt && \
    pip install --no-cache-dir -c /opt/pip-constraints-v4.txt \
        pycolmap==4.1.0 xatlas==0.0.11 geffnet==1.0.2 && \
    python -c "import torch, numpy; assert torch.version.cuda=='11.8', torch.version.cuda; assert numpy.__version__=='1.26.4', numpy.__version__; print('OK: torch y numpy intactos')"

# Código de los DOS modelos, PINEADO por SHA a los commits validados en local.
# (El repo de DSINE YA se movió de commit — fetch por SHA trae EXACTAMENTE lo
# probado, con historia mínima, aunque el proyecto siga cambiando.)
RUN mkdir -p /opt/depth_anything_v2 && cd /opt/depth_anything_v2 && \
    git init -q && git remote add origin https://github.com/DepthAnything/Depth-Anything-V2.git && \
    git fetch -q --depth 1 origin a561b849ebae10a6f5ef49e26c83cbbcd36c71bf && \
    git checkout -q FETCH_HEAD && rm -rf .git assets

RUN mkdir -p /opt/dsine && cd /opt/dsine && \
    git init -q && git remote add origin https://github.com/baegwangbin/DSINE.git && \
    git fetch -q --depth 1 origin ddedde139279ca3a334dbc377a66993ef76fa7dd && \
    git checkout -q FETCH_HEAD && rm -rf .git assets docs paper.pdf

# Checkpoints HORNEADOS (igual que los de MASt3R): se bajan UNA vez aquí,
# gratis en GitHub Actions, y quedan dentro de la imagen.
#  - profundidad (vitb métrico interiores, ~390MB): OBLIGATORIO → si no baja,
#    el build FALLA aquí (mejor enterarse gratis en Actions que en la GPU).
#  - normales DSINE (~291MB): OPCIONAL → si no baja, el worker usa el fallback
#    "normales desde profundidad" y lo avisa en el log del render.
RUN mkdir -p /opt/models && cd /opt/models && \
    wget -q --tries=3 https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth && \
    test -f depth_anything_v2_metric_hypersim_vitb.pth && \
    (wget -q --tries=3 https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt || \
     echo "AVISO: dsine.pt no bajo; el worker usara normales-desde-profundidad") && \
    ls -lh /opt/models/


# ════════════════════════════════════════════════════════════════════════
# PASO 10 (v5.0) — FASE 1: PGSR junto a 2DGS
# ════════════════════════════════════════════════════════════════════════
# POR QUE: medimos que el 52% del ruido de nuestras paredes es ondulacion de
# 20-50 cm. Ni el TSDF ni el aplanado posterior pueden tocar esa banda porque
# se fabrica en el ENTRENAMIENTO: la perdida fotometrica no sabe donde esta
# una pared lisa sin textura y la curva. PGSR impone planaridad durante el
# entrenamiento (profundidad insesgada + consistencia multi-vista fotometrica
# y geometrica), que es exactamente el ataque a esa banda.
#
# 2DGS SE QUEDA INTACTO en /opt/2dgs. El worker elige con MESH_ENGINE:
#     MESH_ENGINE=pgsr  (por defecto)      MESH_ENGINE=2dgs  (respaldo)
# Si PGSR no estuviera en la imagen, el worker lo detecta y usa 2DGS solo.
#
# CONFLICTO DE MODULOS (importante): 2DGS y PGSR tienen paquetes con el MISMO
# nombre (scene/, utils/, arguments/, gaussian_renderer/). No se rompen entre
# si porque Python pone PRIMERO la carpeta del script que se ejecuta: al correr
# /opt/pgsr/train.py ganan los modulos de PGSR aunque /opt/2dgs este en el
# PYTHONPATH. Por eso NO se agrega /opt/pgsr al PYTHONPATH.
#
# LICENCIA — LEER ANTES DE COBRAR POR ESTO: PGSR esta construido sobre el 3DGS
# de INRIA, cuya licencia es de INVESTIGACION, NO COMERCIAL (2DGS viene del
# mismo origen, o sea que esto ya aplicaba antes). Si Vessel Render Lab va a
# ser un producto de pago hay que licenciarlo con INRIA o cambiar el nucleo.
# ════════════════════════════════════════════════════════════════════════

# --recursive es OBLIGATORIO: los submodulos (diff-plane-rasterization,
# simple-knn) son git submodules y sin esto llegan las carpetas VACIAS.
WORKDIR /opt
RUN git clone --recursive https://github.com/zju3dv/PGSR.git pgsr

WORKDIR /opt/pgsr

# Dependencias de PGSR SIN romper las nuestras: su requirements.txt puede
# traer torch/open3d/numpy/opencv y reinstalarlos con otra version, lo que
# romperia todo el post-proceso (dependemos de open3d 0.18 y numpy 1.26.4).
# Por eso se filtran esas lineas antes de instalar.
RUN if [ -f requirements.txt ]; then \
        grep -viE '^[[:space:]]*(torch|torchvision|torchaudio|open3d|numpy|opencv-python|opencv-python-headless)([=<>~!].*)?[[:space:]]*$' \
            requirements.txt > /tmp/pgsr_req.txt || true; \
        echo "--- dependencias de PGSR que SI se instalan ---"; cat /tmp/pgsr_req.txt; \
        pip install --no-cache-dir -r /tmp/pgsr_req.txt || true; \
    else echo "PGSR sin requirements.txt"; fi

# Rasterizador CUDA propio de PGSR. Este es el paso lento del build (5-15 min).
# TORCH_CUDA_ARCH_LIST ya viene de arriba con 8.0;8.6;8.9;9.0 -> cubre A6000
# (8.6) y RTX 4090 (8.9), asi que la misma imagen sirve en los dos pods.
RUN pip install --no-cache-dir --no-build-isolation ./submodules/diff-plane-rasterization

# simple-knn de PGSR: mismo nombre de modulo que el de 2DGS y mismo codigo
# original de 3DGS, asi que reinstalarlo encima es inofensivo.
RUN pip install --no-cache-dir --no-build-isolation ./submodules/simple-knn

# Verificacion de PGSR EN EL BUILD: si algo falla, el build se detiene aqui
# y no te enteras a mitad de un render de una hora.
# OJO (fallo real del primer build): simple_knn._C y los rasterizadores son
# extensiones compiladas CONTRA PyTorch. Si se importan en un proceso donde
# torch NO se importo antes, fallan con:
#     ImportError: libc10.so: cannot open shared object file
# No es que esten mal instalados: es que libc10.so (de torch) no esta cargada
# todavia. Por eso TODO va en UN SOLO python -c que empieza por import torch,
# igual que la verificacion de la v4.2 mas arriba en este mismo Dockerfile.
RUN python -c "import torch; assert torch.version.cuda=='11.8', torch.version.cuda; print('torch', torch.__version__, 'cuda', torch.version.cuda); import diff_plane_rasterization; print('diff-plane-rasterization (PGSR) OK'); import simple_knn._C; print('simple-knn OK'); import diff_surfel_rasterization; print('diff-surfel-rasterization (2DGS) SIGUE OK'); import numpy; assert numpy.__version__=='1.26.4', 'numpy cambiado por PGSR: '+numpy.__version__; import open3d; assert open3d.__version__.startswith('0.18'), 'open3d cambiado por PGSR: '+open3d.__version__; print('numpy y open3d intactos:', numpy.__version__, open3d.__version__)" && \
    test -f /opt/pgsr/train.py && test -f /opt/pgsr/render.py && echo "PGSR train.py y render.py OK" && \
    test -f /opt/2dgs/train.py && echo "2DGS intacto OK" && \
    echo "=== VERIFICACION_PGSR_OK ==="

WORKDIR /opt

# ── Marcador de versión: el worker lo lee y lo imprime en el banner ──
RUN echo "v5.0-pgsr" > /opt/IMAGE_TAG

# ── Paso 9: verificación v4 (imports ligeros, SIN instanciar modelos) ──
# DSINE: sys.path con /opt/dsine PRIMERO para que sus paquetes models/ y utils/
# ganen a los de 2DGS/MASt3R del PYTHONPATH (misma técnica que usa el worker).
RUN python -c "import numpy; assert numpy.__version__=='1.26.4', numpy.__version__; import faiss; import cv2; import open3d; import torch; assert torch.version.cuda=='11.8'; print('re-chequeo post-instalaciones: numpy', numpy.__version__, '+ faiss + cv2 + open3d + torch OK')" && \
    python -c "import pycolmap; print('pycolmap', pycolmap.__version__); import xatlas; print('xatlas OK'); import geffnet; print('geffnet OK')" && \
    python -c "import sys; sys.path.insert(0,'/opt/depth_anything_v2/metric_depth'); from depth_anything_v2.dpt import DepthAnythingV2; print('DAv2 import OK')" && \
    python -c "import sys; sys.path.insert(0,'/opt/dsine'); import geffnet; _o=geffnet.create_model; geffnet.create_model=lambda *a,**k:_o(*a,**{**k,'pretrained':False}); from models.dsine import DSINE; import utils.utils as du; assert hasattr(du,'pad_input'); print('DSINE import OK')" && \
    test -f /opt/models/depth_anything_v2_metric_hypersim_vitb.pth && echo "checkpoint profundidad OK" && \
    echo "=== VERIFICACION_V4_OK ==="

WORKDIR /workspace
CMD ["/bin/bash"]
