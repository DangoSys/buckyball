#!/bin/bash


#######################################
# Wrapper around replace-content.py.
# For a file ($1), write out text ($3) into it
# replacing any area designated by $2.
#######################################
function replace_content
{
  # If BASH_SOURCE is undefined, we may be running under zsh, in that case
  # provide a zsh-compatible alternative
  DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]:-${(%):-%x}}")")"
  file="$1"
  shift
  key="$1"
  shift
  $DIR/replace-content.py "$file" "$key" "$@"
}
