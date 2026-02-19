# syntax=docker/dockerfile:1.7
FROM rust:1.84.1-bookworm

ARG UID=1000
ARG GID=1000
ARG USERNAME=app
ARG UV_VERSION=0.7.2

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CARGO_TERM_COLOR=always

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    time \
    pkg-config \
    build-essential \
    clang \
    cmake \
    python3 \
    python3-venv \
    python3-pip

RUN curl -LsSf "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-installer.sh" | sh \
    && install -m 0755 /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /workspace

COPY Cargo.toml Cargo.lock pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    uv sync --frozen

COPY src ./src
COPY scripts ./scripts
COPY discover_optimal_model.sh ./
COPY README.md ./
COPY tokenizer.json ./

RUN --mount=type=cache,target=/workspace/target \
    --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    chmod +x discover_optimal_model.sh \
    && cargo fetch --locked \
    && cargo build --release --locked \
    && install -m 0755 target/release/whisper_ort_bench /usr/local/bin/whisper_ort_bench

RUN groupadd --gid "${GID}" "${USERNAME}" \
    && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash "${USERNAME}" \
    && usermod -a -G audio "${USERNAME}" \
    && chown -R "${USERNAME}:${USERNAME}" /workspace

USER ${USERNAME}
ENV PATH="/opt/venv/bin:${PATH}"

ENTRYPOINT ["./discover_optimal_model.sh"]
