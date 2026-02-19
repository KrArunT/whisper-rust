FROM rust:1.84.1-bookworm

ARG UID=1000
ARG GID=1000
ARG USERNAME=app
ARG UV_VERSION=0.7.2

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CARGO_TERM_COLOR=always

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    pkg-config \
    build-essential \
    clang \
    cmake \
    python3 \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-installer.sh" | sh \
    && install -m 0755 /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /workspace

COPY Cargo.toml Cargo.lock pyproject.toml uv.lock README.md ./
COPY src ./src
COPY scripts ./scripts
COPY discover_optimal_model.sh ./
COPY audio ./audio
COPY models ./models
COPY results ./results

RUN chmod +x discover_optimal_model.sh \
    && uv sync --frozen \
    && cargo fetch --locked \
    && cargo build --release --locked

RUN groupadd --gid "${GID}" "${USERNAME}" \
    && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash "${USERNAME}" \
    && usermod -a -G audio "${USERNAME}" \
    && chown -R "${USERNAME}:${USERNAME}" /workspace

USER ${USERNAME}
ENV PATH="/workspace/.venv/bin:${PATH}"

ENTRYPOINT ["./discover_optimal_model.sh"]
