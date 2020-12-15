ARG PARENT_IMAGE
ARG PYTORCH_DEPS

# Set base image
FROM $PARENT_IMAGE

# File Author / Maintainer
LABEL Pierre Aumjaud <pierre.aumjaud@ucd.ie>

# Update the repository sources list and install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        libglib2.0-0 \
        ffmpeg \
        freeglut3-dev \
        swig \
        xvfb \
        vim \
        libxrandr2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install anaconda and dependencies
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=3.6 numpy pyyaml scipy ipython mkl mkl-include && \
     /opt/conda/bin/conda install -y pytorch $PYTORCH_DEPS -c pytorch && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

# Install Stable Baselines 3 and dependencies
RUN pip install stable-baselines3[extra,tests,docs]>=0.11.0a2 \
        box2d-py==2.3.8 \
        pybullet \
        gym-minigrid \
        scikit-optimize \
        optuna \
        pytablewriter \
        seaborn \
        pyyaml>=5.1 \
        sb3-contrib>=0.11.0a3 && \
    # Use headless version for docker
    pip uninstall -y opencv-python && \
    pip install opencv-python-headless && \
    rm -rf $HOME/.cache/pip

# Clone rl_reach repository
RUN cd /root/ && \
    git clone https://github.com/PierreExeter/rl_reach.git

# Install custom gym environments
RUN cd /root/rl_reach/gym_envs && \
    pip install -e . && \
    cd /root/rl_reach/

# Set working directory to /root/rl_reach/
WORKDIR /root/rl_reach/

# # Set entrypoint for Gym rendering: commented by Pierre
# COPY docker/entrypoint.sh /tmp/
# RUN chmod +x /tmp/entrypoint.sh
# ENTRYPOINT ["/tmp/entrypoint.sh"]

CMD /bin/bash
