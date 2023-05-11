from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from gym.spaces import Box, Discrete
import numpy as np

class TargetFinderRobo(RobotSupervisorEnv):
    '''
    Simple Robot to find randomly spawning targets
    '''
    def __init__(self):
        super().__init__()
        
        # Observations
        '''
        Observations by index:
            0-1     Robo x, y (-1 to 1 normalized to plane size)
            2       Robo orientation (heading) (-pi to pi)
            3-6    Robo distance sensors (4x single signal value from each sensor) [[8 are possible]]
            7-26   Target x, y positions (10x 2 values from each target. Will set to 0,0 after target is acquired)
        '''
        self.arena = self.getFromDef('ARENA').getField('floorSize')
        aX, aY = self.arena.getSFVec2f()
        min_obs = [-aX/2, -aY/2, -np.pi, 0, 0, 0, 0]
        # Add target positions to observations
        for _ in range(10):
            min_obs.extend([-aX/2, -aY/2])
        min_obs = np.array(min_obs)
        # Configure maximum observation values
        max_obs = -1*np.copy(min_obs)
        # normalized position sensor readings should be ~1
        max_obs[3:7] = [1, 1, 1, 1]
        self.observation_space = Box(low=min_obs, high=max_obs, dtype=np.float64)
        # Actions
        self.action_space = Discrete(4)

        # define the robot
        self.boss = self.getSelf()

        # Make some wheels and initialize
        self.lMotor = self.getDevice('left wheel motor')
        self.rMotor = self.getDevice('right wheel motor')        

        # Episode setup and parameters
        self.steps_per_episode = 2000
        self.episode_score = 0
        self.numSteps = 0
        self.episode_score_list = []
        self.targets_found = 0
        self.obstacleCollisions = -15

        # Environment parameters
        self.arena = self.getFromDef('ARENA').getField('floorSize')
        self.homeBaseRadius = 0.1 # self.getFromDef('BASE').getField('radius').getSFFloat()
        self.target = self.getFromDef('TARGET')
        self.targetField = self.target.getField('translation')

        # Robot controls
        self.rotation_field = self.boss.getField('rotation')
        self.moving = False
        self.posSensors = []

    def get_observations(self):
        obs = []
        obs.extend(self.boss.getPosition()[0:2]) # position x, y
        obs.append(self.boss.getOrientation()[1]) # orientation
        for ps in self.posSensors:
            obs.append(ps.getValue()/2500)
        obs.extend(self.targetField.getSFVec3f()[0:2]) # target position
        obs.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # other targets

        return obs
    
    def targetSpawn(self, t=None, outerBuffer=True):
        maxX, maxY = self.arena.getSFVec2f()
        maxX /= 2
        maxY /= 2
        if outerBuffer:
            maxX *= .95
            maxY *= .95
            
        x = np.random.uniform(-maxX, maxX)
        thresh = np.sqrt( abs(self.homeBaseRadius**2-x**2) ) if abs(x) < self.homeBaseRadius else 0
        y = np.random.uniform(thresh, maxY) * np.random.choice([-1, 1])
        self.targetField.setSFVec3f([x, y, 0.03])

        return x, y
    
    def get_default_observation(self):
        return [0.0 for _ in range(self.observation_space.shape[0])]
    
    def get_reward(self, action=None):
        collisionPenalty = 0

        for target in [self.target]:
            if target.getContactPoints():
                # You can set the target to some outside area here target.getField('translation').setSFVec3f([-1, someUniqueY, 0.03])
                self.targetSpawn()
                self.numSteps = 0
                self.targets_found += 1
                self.episode_score += self.targets_found*2
                return self.targets_found*2

        if len(self.boss.getContactPoints()) > 2:
            self.obstacleCollisions += 1
            if self.obstacleCollisions > 0:
                collisionPenalty = 1
        
        stepReward = -0.002 - 0.01*(collisionPenalty)
        self.episode_score += stepReward
        return stepReward
    
    def is_done(self):
        # end after max # of steps
        if self.numSteps > self.steps_per_episode or self.targets_found > 9:
            print(f"targets found: {self.targets_found}", end="\t")
            self.numSteps = 0
            self.targets_found = 0
            self.episode_score_list.append(self.episode_score)
            return True
     
        
        return False
    
    def solved(self):
        if len(self.episode_score_list) > 100:
            if np.mean(self.episode_score_list[-100:]) > 0.8:
                return True
        return False
    
    def get_info(self):
        return None
    
    def render(self, mode="human"):
        pass

    def turn(self, dir=1):
        self.lMotor.setPosition(float('inf'))
        self.lMotor.setVelocity(3*dir)

        self.rMotor.setPosition(float('inf'))
        self.rMotor.setVelocity(-3*dir)

    def moveForward(self):
        self.lMotor.setPosition(float('inf'))
        self.lMotor.setVelocity(3)
        self.rMotor.setPosition(float('inf'))
        self.rMotor.setVelocity(3)
    
    def noMotion(self):
        self.lMotor.setVelocity(0)
        self.rMotor.setVelocity(0)


    def apply_action(self, action):
        self.numSteps += 1
        self.moving = True

        if(action == 1):
            self.moveForward()
        elif(action == 2):
            self.turn(dir=1) # turn right
        elif(action == 3):
            self.turn(dir=-1) # turn left
        else:
            self.noMotion()
            self.moving = False

    def enablePositionSensors(self, sensorsToUse=[0, 1, 6, 7]):
        '''
        sensors: 0 stars at 1 o' clock position and is labeled sequentially clockwise to 7 at 11 o' clock
            "times" shown below for direction from center, front center of robo is 12:00
            ps0: 1
            ps1: 2
            ps2: 3
            ps3: 5
            ps4: 7
            ps5: 9
            ps6: 10
            ps7: 11
        '''
        if len(self.posSensors) > 0: return

        timestep = super().timestep
        for x in sensorsToUse:
            ps = self.getDevice(f'ps{x}')
            ps.enable(timestep)
            self.posSensors.append(ps)

    def reset(self):
        resetObs = super().reset()
        self.targetSpawn()
        self.enablePositionSensors()
        self.rotation_field.setSFRotation([0.0, 0.0, 1.0, np.random.uniform(-np.pi, np.pi)])
        self.obstacleCollisions = -15
        return resetObs