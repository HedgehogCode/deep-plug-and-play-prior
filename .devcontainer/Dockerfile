FROM tensorflow/tensorflow:2.4.1-gpu
LABEL maintainer="benjamin.wilhelm@uni-konstanz.de"

ARG USERNAME=vsdocker
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Install essentials
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get -y install --no-install-recommends git vim zsh tmux ssh less git-lfs

# Install opencv2 dependency
RUN apt-get -y install --no-install-recommends libgl1-mesa-glx

# Runtime dependencies
COPY ./requirements/requirements_docker.txt /root/requirements_docker.txt
RUN pip install Cython
RUN pip install -r /root/requirements_docker.txt

# Develop dependencies
COPY ./requirements/requirements_dev.txt /root/requirements_dev.txt
RUN pip install -r /root/requirements_dev.txt

# Add jupyter-matplotlib extension
# RUN jupyter labextension install jupyter-matplotlib@0.7.4