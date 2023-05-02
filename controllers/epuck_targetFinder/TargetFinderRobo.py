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
        self.observation_space = Box(low=np.array([-0.5, -0.5, -0.5, -0.5, -np.pi, -np.inf]),
                                     high=np.array([0.5, 0.5, 0.5, 0.5, np.pi, np.inf]),
                                     dtype=np.float64
                                     )
        # Actions
        self.action_space = Discrete(3)

        # define the robot
        self.boss = self.getSelf()

        # Make some wheels and initialize
        self.lMotor = self.getDevice('left wheel motor')
        self.rMotor = self.getDevice('right wheel motor')        

        # Episode setup and parameters
        self.steps_per_episode = 5000
        self.episode_score = 0
        self.numSteps = 0
        self.episode_score_list = []
        self.distanceToTarget = 1
        self.heading = 0
        self.targets_found = 0

        # Robot controls
        self.rotation_field = self.boss.getField('rotation')
        self.target = self.getFromDef('TARGET')
        self.targetField = self.target.getField('translation')

    def get_heading(self):
        return 0

    def get_observations(self):
        xPos = self.boss.getPosition()[0]
        yPos = self.boss.getPosition()[1]
        xTarget = self.targetField.getSFVec3f()[0]
        yTarget = self.targetField.getSFVec3f()[1]
        rot = self.boss.getOrientation()[1]
        self.heading = np.cos(np.arctan2(yPos-yTarget, xPos-xTarget) - rot)
        self.distanceToTarget = np.sqrt((xTarget - xPos)**2 + (yTarget - yPos)**2)

        return [xTarget, yTarget, xPos, yPos, rot, self.distanceToTarget]
    
    def get_default_observation(self):
        return [0.0 for _ in range(self.observation_space.shape[0])]
    
    def get_reward(self, action=None):
        if self.distanceToTarget < 0.1:
            self.targetField.setSFVec3f([np.random.uniform(-0.4, 0.4), np.random.uniform(-0.4, 0.4), 0.03])
            self.numSteps = 0
            self.targets_found += 1
            return 5
        return -0.001 + 0.001*self.heading
    
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


    def apply_action(self, action):
        self.numSteps += 1
        if(action == 1):
            self.turn(dir=1) # turn right
        elif(action == 2):
            self.turn(dir=-1) # turn left
        else:
            self.moveForward()

    def reset(self):
        resetObs = super().reset()
        # self.boss.setOrientation([0, 0, np.pi])
        self.rotation_field.setSFRotation([0.0, 0.0, 1.0, np.random.uniform(-np.pi, np.pi)])
        return resetObs