# for env in Swimmer-v2;
# do
# python plot-mujoco.py --seeds 3 --episodes 4 --title valuedice --legend td3-dicefo-idm-reg,valuedice,value_dicefo,td3-dacfo-ep4,bco-ep4,td3-dac-ep4 --timesteps 500000 --steps 10000 --env $env --shift 0 --plot 1
# done
# 
# for env in HalfCheetah-v2; 
# do
# #python plot-mujoco.py --seeds 5 --episodes 4 --title valuedice --legend td3-dicefo-idm-decay-reg,valuedice,td3-dacfo-ep4,bco-ep4,td3-dac-ep4 --timesteps 500000 --steps 10000 --env $env --shift 0 --plot 0
# python plot-mujoco.py --seeds 3 --episodes 4 --title valuedice --legend td3-dicefo-idm-decay-reg,valuedice,value_dicefo,td3-dacfo-ep4,bco-ep4,td3-dac-ep4 --timesteps 500000 --steps 10000 --env $env --shift 0 --plot 0
# done
# # 
# # 
# #for env in Walker2d-v2 Hopper-v2;
# for env in Hopper-v2;
# do
# python plot-mujoco.py --seeds 3 --episodes 4 --title valuedice --legend td3-dicefo-idm-decay-reg2,valuedice,value_dicefo,td3-dacfo-ep4,bco-ep4,td3-dac-ep4 --timesteps 1010000 --steps 10000 --env $env --shift 0 --plot 0
# done

for env in Humanoid-v2; 
do
#python plot-mujoco.py --seeds 3 --episodes 4 --title valuedice --legend td3-dicefo-idm-decay-reg,valuedice,td3-dacfo-ep4,bco-ep4,td3-dac-ep4 --timesteps 2010000 --steps 10000 --env $env --shift 0 --plot 0
python plot-mujoco.py --seeds 3 --episodes 4 --title valuedice --legend td3-dicefo-idm-decay-reg,valuedice,value_dicefo,td3-dacfo-ep4,bco-ep4,td3-dac-ep4 --timesteps 1010000 --steps 10000 --env $env --shift 0 --plot 0
done


# for env in Ant-v2; 
# do
# python plot-mujoco.py --seeds 3 --episodes 4 --title valuedice --legend td3-dicefo-idm-decay-reg,valuedice,value_dicefo,td3-dacfo-ep4,bco-ep4,td3-dac-ep4 --timesteps 1010000 --steps 10000 --env $env --shift 0 --plot 0
# done

#for env in Hopper-v2;
#do
#python plot-mujoco.py --seeds 3 --episodes 4 --title dacfo --legend td3-dicefo-idm-decay-reg2,td3-dice-ep4,td3-dac-ep4 --timesteps 1010000 --steps 10000 --env $env --shift 0 --plot 1
#done

#for env in InvertedPendulum-v2; 
#do
#python plot-mujoco.py --seeds 5 --episodes 4 --title dacfo --legend td3-dicefo-idm-decay-reg,td3-dacfo-ep4,bco-ep4,td3-dac-ep4 --timesteps 200000 --steps 10000 --env $env --shift 0 --plot 0
#done
#
