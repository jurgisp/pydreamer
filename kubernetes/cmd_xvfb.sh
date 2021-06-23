xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 generator.py $1 --num_steps 10_000_000 --seed $2 --save_to_mlflow
