FROM ubuntu:20.04

# Update default packages
RUN apt-get update

# Get Ubuntu packages
RUN apt-get install -y \
    build-essential \
    curl

# Update new packages
RUN apt-get update -y && apt-get install -y unzip
WORKDIR /opt
ENV LIBTORCH_URL=https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip
RUN curl -fsSL --insecure -o libtorch.zip  $LIBTORCH_URL \
    && unzip -q libtorch.zip \
    && rm libtorch.zip

ENV TORCH_HOME=/opt/libtorch
ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

# Get Rust
WORKDIR /app

COPY . .
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc
ENV PATH="/root/.cargo/bin:${PATH}"

RUN rustup default nightly


EXPOSE 8080

CMD [ "cargo", "run" ]