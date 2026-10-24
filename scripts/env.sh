# Source this in every shell: `source scripts/env.sh`
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$_root/.venv/bin/activate" ] && source "$_root/.venv/bin/activate"
export TINYC_ROOT="$_root"
unset _root
