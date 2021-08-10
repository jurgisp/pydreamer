xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 generator.py --env_id $1 --policy $2 --num_steps $5 --seed $3 --max_steps $4
