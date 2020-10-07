for env in Swimmer-v2;
do
python plot-mujoco.py --seeds 3 --episodes 4 --title explore --legend td3-dicefo-idm-reg,bco-ep4,td3-dac-ep4,trpo-gaifo-ep4,trpo-gail-ep4 --timesteps 1000000 --steps 10000 --env $env --shift 0
done

for env in Ant-v2;
do
python plot-mujoco.py --seeds 3 --episodes 4 --title explore --legend td3-dicefo-idm-decay-reg,bco-ep4,td3-dac-ep4,trpo-gaifo-ep4,trpo-gail-ep4 --timesteps 2000000 --steps 10000 --env $env --shift 0
done

for env in HalfCheetah-v2;
do
python plot-mujoco.py --seeds 3 --episodes 4 --title explore --legend td3-dicefo-idm-decay-reg,bco-ep4,td3-dac-ep4,trpo-gaifo-ep4,trpo-gail-ep4 --timesteps 1000000 --steps 10000 --env $env --shift 0
done

for env in Humanoid-v2;
do
python plot-mujoco.py --seeds 3 --episodes 4 --title explore --legend td3-dicefo-idm-decay-reg,bco-ep4,td3-dac-ep4,trpo-gaifo-ep4,trpo-gail-ep4 --timesteps 2000000 --steps 10000 --env $env --shift 0
done

for env in Hopper-v2 Walker2d-v2;
do
python plot-mujoco.py --seeds 3 --episodes 4 --title explore --legend td3-dicefo-idm-decay-reg2,bco-ep4,td3-dac-ep4,trpo-gaifo-ep4,trpo-gail-ep4 --timesteps 1000000 --steps 10000 --env $env --shift 0
done
