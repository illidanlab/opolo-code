#!/bin/sh

prefix="/mnt/research/judy/self-boosted-rl/baselines"
prefix="/mnt/research-share/projects/zhuzhuan/baselines"
if [ "$#" -le 3 ]
then
    echo "Usage:     ./run_lfrs.sh <env> <algo> <task> <seed> (n-timesteps, optional) <total-jobs, optional> <n-episodes, default1>"
    echo "Example:   ./run_lfrs.sh HalfCheetah-v2 td3 gail4-lfd 1 6 4"
    echo "Example:   ./run_lfrs.sh QbertNoFrameskip ppo2 load 1 6 10"
    echo "Example:   ./run_lfrs.sh PongNoFrameskip ppo2 load 1 6 10"
    exit
else
    env=$1
    algo=$2
    task=$3
    seed=$4
    if [ "$#" -ge 5 ]
	then
	n_job=$5
    fi
    if [ "$#" -ge 6 ]
	then
	n_episodes=$6
    fi
    n_time="-1"
    name=""
    
    logdir="$prefix/$task/$algo/$env/rank$seed"
    test_logdir="$prefix/$task/$algo/$env/rank$seed/test"
    mkdir -p $logdir
    mkdir -p $test_logdir

	program="train_agent.py"
    echo "python $program --env $env --seed $seed --algo $algo --log-dir $logdir --task $task --n-timesteps $n_time --n-jobs $n_job --n-episodes $n_episodes"
    #python $program --env $env --seed $seed --algo $algo --log-dir $logdir --task $task --n-timesteps $n_time --n-jobs $n_job --n-episodes $n_episodes
fi

