# conda exit
# conda deactivate

# path
export PATH=$(echo $PATH | tr ':' '\n' | grep -v buckyball | tr '\n' ':' | sed 's/:$//')
