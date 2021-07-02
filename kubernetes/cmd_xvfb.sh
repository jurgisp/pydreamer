xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 generator.py $1 maze_bouncing_ball --num_steps 20_000_000 --seed $2 --save_to_mlflow --max_steps 2000
