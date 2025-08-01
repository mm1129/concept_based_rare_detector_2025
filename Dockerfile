ARG PYTORCH="2.2.2"
ARG CUDA="12.1"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ARG MMCV="2.1.0"
ARG MMENGINE="0.10.3"
ARG MMDET="3.2.0"
ARG MMDEPLOY="1.3.1"
ARG MMDET3D="1.4.0"
ARG MMPRETRAIN="1.2.0"

ENV CUDA_HOME="/usr/local/cuda" \
    FORCE_CUDA="1" \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.7 8.9+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install apt dependencies
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    curl \
    ffmpeg \
    git \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip dependencies
RUN python3 -m pip --no-cache-dir install \
    aenum \
    nptyping \
    numpy==1.23.5 \
    nvidia-pyindex \
    openmim

RUN python -m pip install pip-tools
# Install mim components
# RUN mim install \
#     mmcv==${MMCV} \
#     mmdeploy==${MMDEPLOY} \
#     mmdet==${MMDET} \
#     mmdet3d==${MMDET3D} \
#     mmengine==${MMENGINE} \
#     mmpretrain[multimodal]==${MMPRETRAIN}


WORKDIR /workspace

COPY projects projects
COPY requirements requirements

#  docker run -it --rm --shm-size=64g  -v $PWD/:/workspace rare2
# --gpus all

# RUN curl -sSL https://install.python-poetry.org | python3 -
# ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml poetry.lock* ./

# RUN poetry install --no-root
# RUN pip install -r requirements/requirements.txt
# Additional pip installations
# RUN pip install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121 #118  # CUDA
# PyTorchのインストール
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121

# transformersの最新バージョンをGitHubからインストール（Qwen2.5-VL対応）
RUN pip uninstall -y transformers && \
    pip install git+https://github.com/huggingface/transformers accelerate>=0.26.0

# qwen-vl-utilsのインストール（Linux向け高速版）
RUN pip install qwen-vl-utils[decord]

# その他の必要なパッケージ
RUN pip install \
    ultralytics==8.0.20 \
    scikit-learn \
    matplotlib \
    spacy \
    openai \
    Pillow \
    python-dotenv \
    nltk \
    regex

# NLTK データのダウンロード（コンテナ内にキャッシュ）
# RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger')"

# Optional: Download spaCy models if noun extraction is needed
RUN python -m spacy download en_core_web_sm
CMD ["python", "projects/detect_rare_images/whole-architecture.py"] # "poetry", "run", 

# python projects/detect_rare_images/whole-architecture.py 
# docker run --gpus all -it --rm --shm-size=64g -v $PWD/:/workspace -v $(realpath ../data_nuscenes/nuscenes):/workspace/data_nuscenes -v /mnt/HDD16TB/dataset/external/nuscenes:/workspace/data/nuscenes qwen bash
# docker run --gpus all -it --rm --shm-size=64g -v $PWD/:/workspace -v $(realpath ../data_nuscenes/nuscenes):/workspace/data_nuscenes blip bash
# rare
# docker run --gpus all -d --shm-size=64g -v $PWD/:/workspace -v $(realpath ../data_nuscenes/nuscenes):/workspace/data_nuscenes qwen bash -c "python projects/detect_rare_images/whole-architecture.py"
# docker run --gpus all -d --shm-size=64g -v $PWD/:/workspace -v $(realpath ../data_nuscenes/nuscenes):/workspace/data_nuscenes blip bash -c "python projects/detect_rare_images/concept_based_rare_detector.py"
# docker exec -it <コンテナID> bash
# qwen

