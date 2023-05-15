
if [ $# -lt 1 ]; then
    echo "Usage: $0 <config-folder>"
    exit 1
fi

# get all yaml file in config folder
# config_files=$(ls $@/*.yaml)
config_files=$(ls $@/*.yaml | sort -r)

for config_file in $config_files
do
    echo "running $config_file"

    # run the experiment
    ./run_exp.sh $config_file

    echo "finished $config_file"
done

echo "Done Experiments"
