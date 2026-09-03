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
  scripts/rsync_pull.sh [options]
  scripts/rsync_pull.sh [options] user@host
  scripts/rsync_pull.sh [options] user@host:port
  scripts/rsync_pull.sh [options] user@host /remote/path/PRISM-VQ
  REMOTE=user@host REMOTE_PATH=/remote/path/PRISM-VQ scripts/rsync_pull.sh [options]

Pull generated run results from a remote server to this local checkout with rsync.

Defaults:
  remote:      ${DEFAULT_REMOTE}
  remote_path: ${DEFAULT_REMOTE_PATH}
  ssh_port:    ${DEFAULT_SSH_PORT:-from SSH config}

Options:
  -p, --port PORT     SSH port.
  -n, --dry-run       Show what would be transferred without changing files.
      --delete        Delete local files in synced paths that do not exist remotely.
      --include-data  Also pull dataset/data/.
      --code          Pull remote code/configs instead of result directories.
      --all           Pull the whole remote project instead of result directories only.
      --skip-newer    Do not overwrite local files that are newer.
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

# Parsed option state. Pull defaults to result-only mode; code and full-project
# pulls require explicit flags.
dry_run=false
delete=false
include_data=false
pull_code=false
pull_all=false
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
    --code)
      pull_code=true
      ;;
    --all)
      pull_all=true
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

# Code-only and full-project pulls are different modes, so keep them exclusive.
if [[ "${pull_code}" == true && "${pull_all}" == true ]]; then
  echo "Choose either --code or --all, not both." >&2
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

# Build the SSH command once and pass it to rsync when customized.
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

# Default pull mode is result-only, so local code is not overwritten by accident.
if [[ "${pull_all}" == false && "${pull_code}" == false ]]; then
  include_opts=(
    --include='/checkpoints/***'
    --include='/outputs/***'
    --include='/res/***'
  )

  if [[ "${include_data}" == true ]]; then
    # Include the parent directory so rsync can descend into dataset/data.
    include_opts+=(
      --include='/dataset/'
      --include='/dataset/data/***'
    )
  fi

  # After the wanted result paths are included, exclude everything else.
  include_opts+=(--exclude='*')
elif [[ "${pull_code}" == true ]]; then
  # Code pull mode excludes generated artifacts unless dataset/data is requested.
  include_opts=()
  exclude_opts+=(
    --exclude='checkpoints/'
    --exclude='outputs/'
    --exclude='res/'
  )

  if [[ "${include_data}" == false ]]; then
    exclude_opts+=(
      --exclude='dataset/data/'
    )
  fi
else
  # Full-project mode relies only on the common cache exclusions above.
  include_opts=()
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

# Make the selected pull mode visible before rsync starts.
if [[ "${pull_all}" == true ]]; then
  echo "Syncing remote project ${REMOTE}:${REMOTE_PATH%/}/ to ${PROJECT_ROOT}/"
elif [[ "${pull_code}" == true ]]; then
  echo "Syncing remote code ${REMOTE}:${REMOTE_PATH%/}/ to ${PROJECT_ROOT}/"
else
  echo "Syncing remote results ${REMOTE}:${REMOTE_PATH%/}/ to ${PROJECT_ROOT}/"
fi
# Trailing slashes sync the project contents, not the parent directory itself.
rsync \
  "${rsync_opts[@]}" \
  "${include_opts[@]}" \
  "${exclude_opts[@]}" \
  "${REMOTE}:${REMOTE_PATH%/}/" \
  "${PROJECT_ROOT}/"
