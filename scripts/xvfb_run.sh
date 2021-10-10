# xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" sh /app/scripts/xhost_run.sh "$@"
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" "$@"
