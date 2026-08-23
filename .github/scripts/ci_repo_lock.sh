#!/usr/bin/env bash
set -euo pipefail

LOCK_DIR=/tmp/buckyball-ci
LOCK_FILE="${LOCK_DIR}/lock"
ACTIVE_SHA_FILE="${LOCK_DIR}/active_sha"
REFCOUNT_FILE="${LOCK_DIR}/refcount"

usage() {
  echo "usage: $0 prepare|enter|exit <sha> <repo> <git-ref>" >&2
  exit 1
}

[[ $# -eq 4 ]] || usage

cmd="$1"
want_sha="$2"
repo="${3/#\~/$HOME}"
git_ref="$4"

mkdir -p "${LOCK_DIR}"
exec 9>"${LOCK_FILE}"

repo_reset() {
  cd "${repo}"
  git fetch --force --prune origin "${git_ref}"
  git checkout --detach FETCH_HEAD
  git reset --hard FETCH_HEAD
  git clean -ffd
  git submodule sync
  git submodule update --init --force
  for submodule in bbdev bebop compiler/thirdparty/buddy-mlir; do
    git -C "${repo}/${submodule}" reset --hard
    git -C "${repo}/${submodule}" clean -ffd
  done
  have_sha="$(git rev-parse HEAD)"
  if [[ "${have_sha}" != "${want_sha}" ]]; then
    echo "ERROR: checkout is ${have_sha}, expected ${want_sha}" >&2
    exit 1
  fi
}

repo_prepare() {
  flock 9
  repo_reset
  echo "${want_sha}" > "${ACTIVE_SHA_FILE}"
  echo 0 > "${REFCOUNT_FILE}"
  flock -u 9
}

repo_enter() {
  while true; do
    flock 9
    active_sha="$(cat "${ACTIVE_SHA_FILE}" 2>/dev/null || true)"
    refcount="$(cat "${REFCOUNT_FILE}" 2>/dev/null || echo 0)"
    have_sha="$(git -C "${repo}" rev-parse HEAD)"
    if [[ "${have_sha}" == "${want_sha}" ]]; then
      echo "${want_sha}" > "${ACTIVE_SHA_FILE}"
      echo $((refcount + 1)) > "${REFCOUNT_FILE}"
      flock -u 9
      return 0
    fi
    if [[ "${active_sha}" == "${want_sha}" ]]; then
      echo $((refcount + 1)) > "${REFCOUNT_FILE}"
      flock -u 9
      return 0
    fi
    if [[ "${refcount}" == "0" ]]; then
      repo_reset
      echo "${want_sha}" > "${ACTIVE_SHA_FILE}"
      echo 1 > "${REFCOUNT_FILE}"
      flock -u 9
      return 0
    fi
    flock -u 9
    echo "waiting: active=${active_sha} refcount=${refcount} want=${want_sha}"
    sleep 10
  done
}

repo_exit() {
  flock 9
  active_sha="$(cat "${ACTIVE_SHA_FILE}" 2>/dev/null || true)"
  refcount="$(cat "${REFCOUNT_FILE}" 2>/dev/null || echo 0)"
  if [[ "${active_sha}" != "${want_sha}" ]]; then
    echo "ERROR: active sha ${active_sha} != job sha ${want_sha}" >&2
    flock -u 9
    exit 1
  fi
  if [[ "${refcount}" -le 0 ]]; then
    echo "ERROR: refcount underflow (${refcount})" >&2
    flock -u 9
    exit 1
  fi
  echo $((refcount - 1)) > "${REFCOUNT_FILE}"
  flock -u 9
}

case "${cmd}" in
  prepare) repo_prepare ;;
  enter) repo_enter ;;
  exit) repo_exit ;;
  *) usage ;;
esac
