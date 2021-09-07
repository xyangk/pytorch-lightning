#!/bin/bash

NUM_HOSTS=`gcloud alpha compute tpus tpu-vm describe $1 --zone europe-west4-a | grep ipAddress | wc -l`
echo "Launching $2 on $1 ($NUM_HOSTS hosts)"

mkdir -p "logs/$1"
for ((i=0;i<=$NUM_HOSTS-1;i++))
do
  cat $2 | gcloud alpha compute tpus tpu-vm ssh $1 --zone europe-west4-a --worker $i &> "logs/$1/worker-$i.log" &
done
wait
