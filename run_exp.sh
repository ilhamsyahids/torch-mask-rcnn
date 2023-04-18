
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config-folder>"
    exit 1
fi

echo "running exp $@"

python3 train.py --config-file $@ > $@.log 2> $@.err < /dev/null
