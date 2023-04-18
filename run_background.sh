
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config-folder>"
    exit 1
fi

source .venv/bin/activate

echo "running $@"

nohup ./run.sh $@ > $@/run.out 2> $@/run.err < /dev/null &
