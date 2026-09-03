#!/usr/bin/env bash
set -euo pipefail

# Default to the local SSH alias and the usual AutoDL project path.
# These can still be overridden by environment variables or CLI arguments.
DEFAULT_REMOTE="${DEFAULT_REMOTE:-autodl-4090d-1}"
DEFAULT_REMOTE_PATH="${DEFAULT_REMOTE_PATH:-/root/autodl-tmp/PRISM-VQ}"
DEFAULT_SSH_PORT="${DEFAULT_SSH_PORT:-}"

# Print usage with the currently resolved defaults.
usage() {
  cat <<USAGE
Usage:
  scripts/rsync_push.sh [options]
  scripts/rsync_push.sh [options] user@host
  scripts/rsync_push.sh [options] user@host:port
  scripts/rsync_push.sh [options] user@host /remote/path/PRISM-VQ
  REMOTE=user@host REMOTE_PATH=/remote/path/PRISM-VQ scripts/rsync_push.sh [options]

Sync local source code to a remote server with rsync.

Defaults:
  remote:      ${DEFAULT_REMOTE}
  remote_path: ${DEFAULT_REMOTE_PATH}
  ssh_port:    ${DEFAULT_SSH_PORT:-from SSH config}

Options:
  -p, --port PORT     SSH port.
  -n, --dry-run       Show what would be transferred without changing files.
      --delete        Delete remote files that do not exist locally.
      --include-data  Also sync generated data/results/checkpoints.
      --skip-newer    Do not overwrite remote files that are newer.
  -h, --help          Show this help message.

Environment:
  REMOTE              SSH target, for example user@host or an SSH config alias.
  REMOTE_PATH         Remote project directory.
  SSH_PORT            Optional SSH port.
  SSH_OPTS            Optional extra ssh options.
  RSYNC_EXTRA_OPTS    Optional extra rsync options.
USAGE
}

# Resolve the repository root from the script location, so the script works
# no matter which directory it is launched from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parsed option state.
dry_run=false
delete=false
include_data=false
skip_newer=false
positional=()
ssh_port_arg=""

# Manual argument parsing keeps the script dependency-free and portable.
while (($#)); do
  case "$1" in
    -p|--port)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1." >&2
        usage >&2
        exit 2
      fi
      ssh_port_arg="$2"
      shift
      ;;
    -n|--dry-run)
      dry_run=true
      ;;
    --delete)
      delete=true
      ;;
    --include-data)
      include_data=true
      ;;
    --skip-newer)
      skip_newer=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while (($#)); do
        positional+=("$1")
        shift
      done
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      positional+=("$1")
      ;;
  esac
  shift
done

# Positional arguments override environment variables, which override defaults.
REMOTE="${positional[0]:-${REMOTE:-${DEFAULT_REMOTE}}}"
REMOTE_PATH="${positional[1]:-${REMOTE_PATH:-${DEFAULT_REMOTE_PATH}}}"
remote_port=""

# Allow a convenient user@host:port shorthand.
if [[ "${REMOTE}" =~ ^(.+):([0-9]+)$ ]]; then
  REMOTE="${BASH_REMATCH[1]}"
  remote_port="${BASH_REMATCH[2]}"
fi

# Port precedence: --port, then user@host:port, then SSH_PORT, then default.
ssh_port="${ssh_port_arg:-${remote_port:-${SSH_PORT:-${DEFAULT_SSH_PORT}}}}"

if [[ -z "${REMOTE}" || -z "${REMOTE_PATH}" ]]; then
  echo "REMOTE and REMOTE_PATH are required." >&2
  usage >&2
  exit 2
fi

# Shared rsync behavior: preserve metadata, show progress, and keep partial
# transfers so interrupted syncs can resume more cheaply.
rsync_opts=(
  --archive
  --human-readable
  --info=progress2
  --partial
)

# Build the SSH command once and reuse it for both mkdir and rsync transport.
ssh_cmd=(ssh)
if [[ -n "${ssh_port}" ]]; then
  ssh_cmd+=(-p "${ssh_port}")
fi

# Allow advanced SSH options such as StrictHostKeyChecking or IdentityFile.
if [[ -n "${SSH_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  ssh_extra_opts=(${SSH_OPTS})
  ssh_cmd+=("${ssh_extra_opts[@]}")
fi

# Always ignore VCS metadata, Python caches, virtualenvs, and local tool caches.
exclude_opts=(
  --exclude='.git/'
  --exclude='.pytest_cache/'
  --exclude='__pycache__/'
  --exclude='*.py[cod]'
  --exclude='.venv/'
  --exclude='venv/'
  --exclude='.mypy_cache/'
  --exclude='.ruff_cache/'
  --exclude='.cache/'
)

# Push is code-focused by default; large generated artifacts are opt-in.
if [[ "${include_data}" == false ]]; then
  exclude_opts+=(
    --exclude='checkpoints/'
    --exclude='outputs/'
    --exclude='res/'
    --exclude='dataset/data/'
  )
fi

# Destructive or protective behaviors must be enabled explicitly.
if [[ "${dry_run}" == true ]]; then
  rsync_opts+=(--dry-run)
fi

if [[ "${delete}" == true ]]; then
  rsync_opts+=(--delete)
fi

if [[ "${skip_newer}" == true ]]; then
  rsync_opts+=(--update)
fi

# Tell rsync to use the custom SSH command only when one was configured.
if [[ "${#ssh_cmd[@]}" -gt 1 ]]; then
  rsync_opts+=(-e "$(printf '%q ' "${ssh_cmd[@]}")")
fi

# Extra rsync flags are appended last so callers can tune special cases.
if [[ -n "${RSYNC_EXTRA_OPTS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_opts=(${RSYNC_EXTRA_OPTS})
  rsync_opts+=("${extra_opts[@]}")
fi

# Ensure the target directory exists for real pushes. Dry-runs stay read-only.
if [[ "${dry_run}" == false ]]; then
  "${ssh_cmd[@]}" "${REMOTE}" "mkdir -p $(printf '%q' "${REMOTE_PATH}")"
fi

# Trailing slashes sync the project contents, not the parent directory itself.
echo "Syncing local ${PROJECT_ROOT}/ to ${REMOTE}:${REMOTE_PATH%/}/"
rsync \
  "${rsync_opts[@]}" \
  "${exclude_opts[@]}" \
  "${PROJECT_ROOT}/" \
  "${REMOTE}:${REMOTE_PATH%/}/"
