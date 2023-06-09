
# How to run

```bash
nohup ./run.sh {config_folder} > run.out 2> run.err < /dev/null &
```

Explanation:

```bash
nohup myscript.sh > myscript.out 2> &1 < /dev/null &
#\__/               \___________/ \__/ \________/  ^
#|                    |          |      |          |
#|                    |          |      |          run in background
#|                    |          |      |
#|                    |          |      don't expect input
#|                    |          |
#|                    |          redirect stderr to stdout or use script.err instead to write into file
#|                    |
#|                    redirect stdout to myscript.log
#|
#keep the command running
#no matter whether the connection is lost or you logout
```
