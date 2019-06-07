# TD3-BipedalWalker-v2-PyTorch

PyTorch implementation of [Twin Delayed DDPG](https://arxiv.org/abs/1802.09477) (TD3) tested on the following environments:

- [Bipedal Walker v2](http://gym.openai.com/envs/BipedalWalker-v2/) 

- [Lunar Lander Continuous v2](http://gym.openai.com/envs/LunarLanderContinuous-v2/) 

- [Walker2d v1 (Roboschool)](https://github.com/openai/roboschool)

- [HalfCheetah v1 (Roboschool)](https://github.com/openai/roboschool)

### Usage
- To test a preTrained network : run `test.py`

- To train a new network : run `train.py`

## Dependencies
Trained and tested on:
```
Python 3.6
PyTorch 0.4.1
NumPy 1.15.3
gym 0.10.8
Roboschool 1.0.46
Pillow 5.3.0
```

## Results

BipedalWalker-v2 (800 episodes)            |  LunarLanderContinuous-v2 (1500 episodes)
:-------------------------:|:-------------------------:
![](https://github.com/nikhilbarhate99/TD3-BipedalWalker-v2-PyTorch/blob/master/gif/GIF-ONE.gif)  |  ![](https://github.com/nikhilbarhate99/TD3-PyTorch-BipedalWalker-v2/blob/master/gif/GIF-TWO.gif)   |


RoboschoolWalker2d-v1 (lr=0.002, 1400 episodes)            |  HalfCheetah-v1 (lr=0.002, 1400 episodes)
:-------------------------:|:-------------------------:
![](https://github.com/nikhilbarhate99/TD3-BipedalWalker-v2-PyTorch/blob/master/gif/GIF-FOUR.gif)  |  ![](https://github.com/nikhilbarhate99/TD3-PyTorch-BipedalWalker-v2/blob/master/gif/GIF-FIVE.gif)   |


*The results are not consistent for BipedalWalker-v2 env

## References

- Official [TD3 paper](https://arxiv.org/abs/1802.09477) and [code](https://github.com/sfujim/TD3)


- [OpenAI SpinningUp in Deep RL](https://spinningup.openai.com/en/latest/algorithms/td3.html)

