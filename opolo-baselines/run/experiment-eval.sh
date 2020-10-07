#for env in HalfCheetah-v2 Humanoid-v2 Ant-v2;
#do
##python eval-mujoco.py --env $env --legends eval-td3-dicefo-idm-decay-reg --algo td3dicefo --seeds 3 --episodes 4 
#python eval-mujoco.py --env $env --legends eval-td3-dac-ep4 --algo td3dac --seeds 3 --episodes 4 
#done
#
# for env in Swimmer-v2
# do
# python eval-mujoco.py --env $env --legends eval-td3-dicefo-idm-reg --algo td3dicefo --seeds 3 --episodes 4 
# python eval-mujoco.py --env $env --legends eval-bco-ep4 --algo td3bco --seeds 3 --episodes 4 
# python eval-mujoco.py --env $env --legends eval-td3-dac-ep4 --algo td3dac --seeds 3 --episodes 4 
# done
#
#for env in Hopper-v2 Walker2d-v2
#do
#python eval-mujoco.py --env $env --legends eval-td3-dac-ep4 --algo td3dac --seeds 3 --episodes 4 
#done

#for env in HalfCheetah-v2 Hopper-v2 Walker2d-v2 Swimmer-v2 Ant-v2 InvertedPendulum-v2; #Humanoid-v2 
for env in Humanoid-v2; 
#for env in HalfCheetah-v2 Ant-v2;
#for env in Ant-v2;
#for env in Hopper-v2 Walker2d-v2;
#for env in Hopper-v2;
#for env in Swimmer-v2; 
#for env in InvertedPendulum-v2;
do
#./run_multi.sh $env td3dacfo eval-td3-dacfo-ep4 0 6 4    
#./run_multi.sh $env trpogaifo eval-trpo-gaifo-ep4 0 6 4    
#./run_multi.sh $env td3bco eval-bco-ep4 0 6 4    
#./run_multi.sh $env td3dac eval-td3-dac-ep4 0 6 4    
#./run_multi.sh $env td3dacfo eval-td3-dacfo-ep4 0 6 4    
#./run_multi.sh $env td3dicefo eval-td3-dicefo-idm-decay-reg 0 6 4    
#./run_multi.sh $env td3dicefo eval-td3-dicefo-idm-decay-reg2 3 6 4    
#./run_multi.sh $env td3dicefo eval-td3-dicefo-idm-reg 3 6 4    
#./run_multi.sh $env trpogail eval-trpo-gail-ep4 0 6 4    
#./run_multi.sh $env td3dicefo eval-td3-dicefo-ep4 0 6 4    

python eval-mujoco.py --env $env --legends eval-td3-dacfo-ep4 --algo td3dacfo --seeds 5 --episodes 4 --shift 0 
#python eval-mujoco.py --env $env --legends eval-trpo-gaifo-ep4 --algo trpogaifo --seeds 3 --episodes 4 
#python eval-mujoco.py --env $env --legends eval-trpo-gail-ep4 --algo trpogail --seeds 3 --episodes 4 
#python eval-mujoco.py --env $env --legends eval-bco-ep4 --algo td3bco --seeds 3 --episodes 4 --shift 1 
#python eval-mujoco.py --env $env --legends eval-td3-dicefo-ep4 --algo td3dicefo --seeds 5 --episodes 4 
#python eval-mujoco.py --env $env --legends eval-td3-dac-ep4 --algo td3dac --seeds 5 --episodes 4 --shift 0 
#python eval-mujoco.py --env $env --legends eval-td3-dicefo-idm-decay-reg --algo td3dicefo --seeds 5 --episodes 4 --shift 0
#python eval-mujoco.py --env $env --legends eval-td3-dicefo-idm-decay-reg2 --algo td3dicefo --seeds 5 --episodes 4
#python eval-mujoco.py --env $env --legends eval-td3-dicefo-idm-reg --algo td3dicefo --seeds 3 --episodes 4 
done
