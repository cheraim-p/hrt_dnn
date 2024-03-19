#build docker from dockerfile
$nohup docker build -f Dockerfile -t ich/ich-docker:v2.1  . < /dev/null > building.log 2>&1 &
$cat building.log

#run docker
$docker run --gpus device=1 -it -v /mnt/ich/in:/host ich/ich-docker:v2.1

#predict 
$ python ich_main.py input_dir output_dir
