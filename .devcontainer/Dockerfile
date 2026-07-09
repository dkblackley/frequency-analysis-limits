FROM rust:1.95-bookworm

RUN apt-get update && \
    apt-get install -y \
    cmake \
    clang \
    libclang-dev \
    build-essential \
    protobuf-compiler \
    libprotobuf-dev \
    coinor-libcbc-dev \
    librsvg2-bin \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*