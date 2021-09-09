xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" bash /app/scripts/xhost_run.sh "$@"
