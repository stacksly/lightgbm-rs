FROM rust:1.71.1
RUN apt update
RUN apt install -y cmake libclang-dev libc++-dev gcc-multilib
WORKDIR /app
