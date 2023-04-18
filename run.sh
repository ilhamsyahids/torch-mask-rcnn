
# get all yaml file in config folder
config_files=$(ls $@/*.yaml)

for config_file in $config_files
do
    echo "running $config_file"

    # run the experiment
    ./run_exp.sh $config_file

    echo "finished $config_file"
done
