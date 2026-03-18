From蘑菇书+王树森：

核心目标：找到最佳策略π（是个概率分布，具有随机性，避免可预判而失效），或者最佳动作价值函数Q_star

### 基本概念
State space: set of S(s_i)状态

Action space: set of A (a_i)行为

Reward: 是数值，postive代表鼓励，negative表示惩罚。依赖于当前的state和action，而不是下一个state（eg.撞墙弹回原来位置和保持不动这两个action的下一个state都是原先位置，但要给前者惩罚）

Trajectory: （state - action - reward） chain

Return: 一个trajectory链上所有的reward总和

Episode：从start到target1次就是1个episode

---

**PDF概率密度函数: **Policy ![image](https://cdn.nlark.com/yuque/__latex/b2d0aa4b5201a2d0c5eda4ce212ab71c.svg)

有**两个随机性**：action -![image](https://cdn.nlark.com/yuque/__latex/39b826045f016b5963c06933a5546cdc.svg)决定 ;  s' 状态转移函数 -![image](https://cdn.nlark.com/yuque/__latex/aa10ab79e598dd194f3039b09e838e0c.svg)。第一个随机性来自agent采取策略的动作随机性，第二个随机性来自环境接受到agent动作后根据状态转移概率用随机抽样来得到下一个状态。



**Discounted return**:![image](https://cdn.nlark.com/yuque/__latex/c5ac69de37823b244660af9ce1c8d5dc.svg),每个R都是随机变量，取决于前一个a和p。![image](https://cdn.nlark.com/yuque/__latex/37713bea5bc8e384deb3945aeca2b979.svg)是权重，距离最近的观测权重最高，介于0-1，是超参数，需要自己调。![image](https://cdn.nlark.com/yuque/__latex/63e0eb2fed6a732352dedc8a1e22ec7c.svg)。目标是让（长期）Ut越大越好。

**Action-Value function**： ![image](https://cdn.nlark.com/yuque/__latex/5fc954190400be525dcb0a9f834de6aa.svg)和![image](https://cdn.nlark.com/yuque/__latex/f962ea99e6cfc66ffa55d09a07306003.svg)都有关，是Ut(未来所有奖励的加权求和)的条件期望，期望把未来的s和a都积分积掉，只留下st和at。

![image](https://cdn.nlark.com/yuque/__latex/831ec43c319077b21a2fb0f5443ccd1c.svg): 基于观测到的state，选择最佳action（Qπ给a动作打分）

**State-Value function**:![image](https://cdn.nlark.com/yuque/__latex/cd8adc81cc9f573524e402eb82437130.svg)

Vπ是Qπ的期望，评估当前局势，把A积分积掉，与动作a无关; 

对于给定π，可以评估当前状态好坏；

通过积s求V期望，可以评估π的好坏

**Markov decision process（MDP）**: 

M-马尔可夫性质-memoryless：条件概率可以忽略历史，只关注![image](https://cdn.nlark.com/yuque/__latex/2509d24a153b161a62f5d394150dc916.svg)，<font style="color:rgba(0, 0, 0, 0.86);background-color:rgba(255, 255, 255, 0.9);">未来只依赖于当前状态</font>

d-就是Policy

p-就是几个possibility



### Value-based价值学习
#### DQN（Deep Q-Network）	
就是用神经网络近似Q*，基于价值学习

Value network（critic）：![image](https://cdn.nlark.com/yuque/__latex/8e5ff11bef169e84cea5e44ef5fbea40.svg)

![image](https://cdn.nlark.com/yuque/__latex/eebf518d62b5917c86a22c045016d0f2.svg)输入是s，输出是对每个a的打分，w是神经网络的参数

#### TD算法（Temporal Difference Learning） 
（用来训练DQN，用奖励来更新模型参数）

观测到st和at，

预测到![image](https://cdn.nlark.com/yuque/__latex/a813d73513240168c5c0973d5e80cb6e.svg)，输出qt是对a的打分

用反向传播对DQN求导，得到梯度

环境提供新的st+1和rt

> 根据Q的意义是求E（Ut），Ut的性质推得的
>

计算TD target：![image](https://cdn.nlark.com/yuque/__latex/da761807cd7d15888fcf9dc44ef28ede.svg)

 ![image](https://cdn.nlark.com/yuque/__latex/5849b510b8fc0a4f478238b00ef9818d.svg)，再做梯度下降减小loss

![image](https://cdn.nlark.com/yuque/__latex/c3bb00bfae458481b654732ee8118ee3.svg)

### Policy-based策略学习
Policy π(a|s)是概率密度函数，输入是s，输出是a的概率分布，agent进行随机抽样得到动作a。

Policy network（actor）：![image](https://cdn.nlark.com/yuque/__latex/90a35acc41264879658bf4a412843b7a.svg)

![image](https://cdn.nlark.com/yuque/__latex/1f9de4634841aad6453d79eaaa15023d.svg)

![image](https://cdn.nlark.com/yuque/__latex/0da3c4c8e43b207e91b79c08599fd9e2.svg)

#### Policy gradient ascent
![image](https://cdn.nlark.com/yuque/__latex/de95d0ecc7732f6eb254a831c5bf7554.svg)

![image](https://cdn.nlark.com/yuque/__latex/12c9c211545756976d194759aab4cb3e.svg)

![image](https://cdn.nlark.com/yuque/__latex/3e99a433b5ce93e5384adf8232366000.svg)蒙特卡洛近似，适用于连续情况

### Actor- Critic Method
**State-Value function**:![image](https://cdn.nlark.com/yuque/__latex/b56f2f663952335406ddb11edce51d1c.svg)

思想：构造Value network和Policy network来近似两个函数，再用actor-critic方法同时学习两个网络

Policy network（actor）：![image](https://cdn.nlark.com/yuque/__latex/49bc4950abe6c381099f549d41cd8013.svg)，策略控制运动，像运动员

![画板](https://cdn.nlark.com/yuque/0/2025/jpeg/59026724/1753586618950-c0d195a9-f05a-4284-8a7b-8d41a3fe9334.jpeg)

Value network（critic）：![image](https://cdn.nlark.com/yuque/__latex/8e5ff11bef169e84cea5e44ef5fbea40.svg)，给运动打分，像裁判

![画板](https://cdn.nlark.com/yuque/0/2025/jpeg/59026724/1753587019220-2fb3faed-38dc-4da4-9141-1e0bdaeccc1d.jpeg)

Train: 

+ 更新policy network通过critic，即![image](https://cdn.nlark.com/yuque/__latex/ed5a4aa5e092e303a69c608582c70db9.svg)，为了提升V的大小，即提升打分
+ 更新value network通过**TD算法**环境奖励反馈，即![image](https://cdn.nlark.com/yuque/__latex/c9b08ae6d9fed72562880f75720531bc.svg)，为了让打分更加精准



#### AlphaGo应用实例
首先是用behavior cloning训练策略，本质是模仿学习，和强化学习的区别在于不获得奖励反馈，只是模仿专家行为（16万局高段位对局棋谱）形成策略。局限点在于只能根据见过的state做出回应，没见过的state（招式）会表现的非常糟糕。进一步用强化学习训练可以弥补这个缺陷，因为会探索更多state，并且可以根据奖励知道应该做什么action。设置player和opponent，用policy gradient优化策略网络，再单独训练一个价值网络（先用策略网络做自我博弈）。实战用蒙特卡洛树搜索：策略函数排除不好的action，选出可能的action，再用策略函数假想对手的应招做模拟进行打分（包括评估state-value得分v和模拟下完整盘棋的r，然后把![image](https://cdn.nlark.com/yuque/__latex/1f9059e59398f18cdbb5da6f3d3a9fa5.svg)作为对a的打分，进一步更新action-values），选择得分最高的action



### 蒙特卡洛算法
随机算法：通过随机样本来估算真实值，不精确但够用

应用1:近似积分

一元函数：先随机抽n个样本，计算函数值，取平均，在乘以区间长度，记为Qn，就是近似积分值。

![](https://cdn.nlark.com/yuque/0/2025/png/59026724/1754191619604-02f41e22-001b-488c-b3f9-90b6c57f5b36.png)

应用2:近似期望

X：一维随机变量，p(x)：概率密度函数

先根据p(x)随机抽样，计算函数值，取平均，记为Qn

![](https://cdn.nlark.com/yuque/0/2025/png/59026724/1754191896416-37d930ea-e756-416d-99a8-d8d8cca165ed.png)

### Experience Replay
目标：重复利用之前的信息经验，并把信息经验打散消除相关性。提升DQN的表现的标准技巧

A transition：![image](https://cdn.nlark.com/yuque/__latex/da9797f6517cf32cbc963fe127d84040.svg)就是经验

![](https://cdn.nlark.com/yuque/0/2025/png/59026724/1754828774453-1ce4006d-de16-4f29-9a3e-a5b503a3296a.png)

n是个超参数，通常设置在百万级。如果存入Buffer数组的经验数超过n，就把最老的替换掉。

沿用TD算法，每次从buffer抽{mini-batch}个transition![image](https://cdn.nlark.com/yuque/__latex/da9797f6517cf32cbc963fe127d84040.svg)，计算TD error，SGD更新w

#### Prioritized Experience Replay
思想：用非均匀抽样代替均匀抽样

少出现的transition更重要，一般用TD error判断，其绝对值越大（说明不熟悉），就更重要，给更大的权重（抽样概率）。

需要相应调整学习率，抵消掉不同权重带来的偏差（权重越大，学习率越小）

![](https://cdn.nlark.com/yuque/0/2025/png/59026724/1754829519491-c481fffb-1dd2-486a-82c3-dd28760a847d.png)

### 高估问题
bootstrap：“自己把自己举起来”，用一个估算去更新其他估算（TD算法用![image](https://cdn.nlark.com/yuque/__latex/776dbac38978dc7298a7cae99bbd8516.svg)更新t时刻的估计）

高估原因：1. TD公式由max操作（因为估值相当于随机增加均值为0的noise，最大估值的平均值肯定大于等于实际动作价值）。 2. 当前高估导致bootstrap更高估。

非均匀的动作价值高估会带来误导，做出非最佳动作。

#### Target Network
用另外一个target network代替TD target，避免自举，缓解bootstrap带来的高估问题（但不完全，因为![image](https://cdn.nlark.com/yuque/__latex/f6d13cb53d7f8d53ff19f4c7c2a3bcd3.svg)依赖于w）

![](https://cdn.nlark.com/yuque/0/2025/png/59026724/1754877994855-341bd301-9026-4e94-9d51-a10223a9cc32.png)

![](https://cdn.nlark.com/yuque/0/2025/png/59026724/1754878498037-c1ea48d2-2b1b-486d-bd9e-a833d50263ed.png)

#### Double DQN
缓解最大化造成的高估。关键在于selection和evaluation。让“选动作”和“评价值”分开，减少高估

![](https://cdn.nlark.com/yuque/0/2025/png/59026724/1754878963855-9e4fc8a8-d8d6-421b-b17a-658c2ce54a65.png)

### Dueling Network
 **Optimal advantage function = Optimal action-value function** -** Optimal state-value function :**![image](https://cdn.nlark.com/yuque/__latex/da0deac5f38ad13ff0f134e24afedaf4.svg)意义是动作a相对于baseline的优势

根据![image](https://cdn.nlark.com/yuque/__latex/c7bb7755abdac094b821110cc38926e2.svg)![image](https://cdn.nlark.com/yuque/__latex/3e489aa264bd92be8acbc2c544ff167f.svg)

变形得到![image](https://cdn.nlark.com/yuque/__latex/0282391852c2d582150ffc744ec7d0fe.svg)加上max这一项可以避免不唯一性值（V和A的上下波动导致）

![](https://cdn.nlark.com/yuque/0/2025/png/59026724/1754882211263-7074a61c-1b38-4344-888f-27f9316c6e2a.png)![](https://cdn.nlark.com/yuque/0/2025/png/59026724/1754882246891-8231eb3d-93dc-467a-a21b-ac4638e3c6a0.png)

由此可以用搭建的dueling network近似Q*(s,a)

![image](https://cdn.nlark.com/yuque/__latex/cd77bb6006f5c55920d8942060034cfb.svg)

![](https://cdn.nlark.com/yuque/0/2025/png/59026724/1754882691675-32142917-0ad6-48a1-9cd4-fcafa9ddb427.png)![](https://cdn.nlark.com/yuque/0/2025/png/59026724/1754882768630-dba300cc-6bc8-44f0-82f1-67277f84e5c9.png)

Training 继续用TD算法



### 策略梯度中的baseline


