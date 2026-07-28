ARG NIX_BASE_IMAGE=docker.io/nixos/nix:latest
FROM ${NIX_BASE_IMAGE}

ARG GIT_REMOTE
ARG GIT_COMMIT

LABEL org.opencontainers.image.source="https://github.com/DangoSys/buckyball"

RUN mkdir -p /etc/nix && \
    printf '%s\n' \
      'experimental-features = nix-command flakes' \
      'accept-flake-config = true' \
      'sandbox = false' \
      'max-jobs = auto' \
      'cores = 0' \
      >> /etc/nix/nix.conf

ENV NIX_CONFIG="experimental-features = nix-command flakes"

WORKDIR /workspace
RUN nix --extra-experimental-features "nix-command flakes" shell nixpkgs#git -c bash -lc '\
      git init buckyball && \
      cd buckyball && \
      git remote add origin "$GIT_REMOTE" && \
      git fetch --depth 1 origin "$GIT_COMMIT" && \
      git checkout FETCH_HEAD && \
      scripts/nix/download.sh \
    '

WORKDIR /workspace/buckyball

RUN nix --extra-experimental-features "nix-command flakes" build
RUN NIX_QUIET=1 nix --extra-experimental-features "nix-command flakes" develop -c true

CMD ["nix", "--extra-experimental-features", "nix-command flakes", "develop", "-c", "bash"]
