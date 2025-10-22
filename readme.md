To run on linux/wsl:

```bash
docker build -t connect-four-rl .
xhost +local:
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix connect-four-rl
```