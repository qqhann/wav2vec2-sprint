FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ARG PYTHON_VERSION=3.8

ARG PYTHON_VERSION=3.8
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  git \
  curl \
  ca-certificates \
  libjpeg-dev \
  libpng-dev \
  git-lfs \
  sox && \
  rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
  chmod +x ~/miniconda.sh && \
  ~/miniconda.sh -b -p /opt/conda && \
  rm ~/miniconda.sh && \
  /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
  /opt/conda/bin/conda install -y pytorch==1.6.0 torchvision torchaudio==0.6.0 cudatoolkit=10.2 -c pytorch
ENV PATH /opt/conda/bin:$PATH

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
    datasets \
    jiwer==2.2.0 \
    soundfile \
    torchaudio \
    lang-trans==0.6.0 \
    librosa==0.8.0

RUN pip uninstall -y typing allennlp

RUN pip install git+https://github.com/huggingface/transformers.git

RUN mkdir -p /workspace/wav2vec/

COPY fine-tune-xlsr-wav2vec2-on-turkish-asr-with-transformers.ipynb finetune.sh run_common_voice.py  finetune_with_params.sh /workspace/wav2vec/

COPY home-server.html run_all.sh /usr/bin/

RUN chown -R 42420:42420 /workspace

RUN chown -R 42420:42420 /usr/bin/run_all.sh

#Default training env variables
ENV model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
    dataset_config_name="clean" \
    output_dir="/opt/ml/checkpoints" \
    cache_dir="/workspace/data" \
    num_train_epochs="1" \
    per_device_train_batch_size="32" \
    per_device_eval_batch_size="32" \
    evaluation_strategy="steps" \
    learning_rate="3e-4" \
    warmup_steps="500" \
    save_steps="10" \
    eval_steps="10" \
    save_total_limit="1" \
    logging_steps="10" \
    feat_proj_dropout="0.0" \
    layerdrop="0.1" \
    max_train_samples=100 \
    max_val_samples=100


# huggingfaceの認証情報
RUN mkdir -p /root/.huggingface
RUN echo -n CsggHkrDoVAVAgEzvgozxQLbDlardRcuexSSQhSAFZTuRnLXPBupdlyRKfKbKKHSsWtDLeCMiTcmBWBTtGIcIiyOHBItTPnZNEKXOdPFxbEdCgiwWHiSuhefFWKOOVmv > /root/.huggingface/token

WORKDIR /workspace
#ENTRYPOINT []
#CMD ["sh", "/usr/bin/run_all.sh"]
