import os
import rclpy
from rclpy.node import Node
from rclpy.qos import ReliabilityPolicy, QoSProfile

from rclpy.qos import qos_profile_sensor_data

from rclpy.executors import Executor
from rclpy.executors import MultiThreadedExecutor
from rclpy.executors import SingleThreadedExecutor


from std_msgs.msg import String
from geometry_msgs.msg import Pose

from custom_interfaces.srv import BimanualJson
from custom_interfaces.srv import UserInput
from robotiq_3f_gripper_ros2_interfaces.srv import Robotiq3FGripperOutputService
from robotiq_3f_gripper_ros2_interfaces.msg import Robotiq3FGripperInputRegisters

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import SystemMessage

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI

from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import MessagesPlaceholder


import time
import json

from typing import Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field



#global var to store robot state 
robot_full_pose = {"step_1":{   "left_ee_coor"           : [0.5,0.3,0.5],
                                "right_ee_coor"          : [-0.5,0.3,0.5],
                                "left_ee_orientation"    : [1.0,0.0,1.0,0.0],
                                "right_ee_orientation"   : [1.0,0.0,1.0,0.0],
                                "left_gripper_state"     : True,
                                "right_gripper_state"    : True}}

#global var to store the item dict
item_dict_global = None




def send_pose_to_robots(pose: dict):
    
    pose_str = json.dumps(pose)
    planner_client = PlannerClient()
    #planner_client.get_logger().info(f"sending pose str: {pose_str}")
    result = planner_client.send_request(pose_str)
    planner_client.destroy_node()
    return result

#------------------------functions-to-work-as-llm-tools----------------------

def move_single_arm(side:str, coordinates, orientation):
    new_pose = robot_full_pose
    
    if side == 'right':
        new_pose["step_1"]["right_ee_coor"] = coordinates
        new_pose["step_1"]["right_ee_orientation"] = orientation
    elif side == 'left':
        new_pose["step_1"]["left_ee_coor"] = coordinates
        new_pose["step_1"]["left_ee_orientation"] = orientation
    result = send_pose_to_robots(new_pose)
    return [result.success, result.msg]



def move_both_arms(left_coordinates, left_orientation, right_coordinates, right_orientation):
    new_pose = robot_full_pose
    
    new_pose["step_1"]["right_ee_coor"] = right_coordinates
    new_pose["step_1"]["right_ee_orientation"] = right_orientation
    new_pose["step_1"]["left_ee_coor"] = left_coordinates
    new_pose["step_1"]["left_ee_orientation"] = left_orientation
    result = send_pose_to_robots(new_pose)
    return [result.success, result.msg]

def use_gripper(side, state):
    if side == 'left':
        gripper_client = GripperClient()
        #gripper_client.get_logger().info("activating left gripper")
        gripper_client.send_request(state)
        gripper_client.destroy_node()

def get_item_dict(empty = None):
    return item_dict_global


def get_full_robot_pose(empty = None):
    return robot_full_pose

def get_pre_grasp_pose(object_position):
    op = object_position
    pose = {'end_effector_position': [op[0], op[1], op[2]+0.2],
            'end_effector_orientation':[1,0,1,0]} #fix this later
    return pose

def get_grasp_pose(object_position):
    op = object_position
    pose = {'end_effector_position' : op,
            'end_effector_orientation': [1,0,1,0]} #fix this later 
    return pose 


def grasp_object(side, object_position, grasp_orientation):
    use_gripper(side,False) #ensure the gripper is open
    
    #select a pre defined grasping orientation 
    if grasp_orientation == 'horizontal' and side == 'left':
        new_orientation = [0.004814, -0.02624, -0.99963, 0.0049]
    elif grasp_orientation == 'vertical' and side == 'left':
        new_orientation = [-0.7, 0.0, -0.7, 0.0]
    
    elif grasp_orientation == 'horizontal' and side == 'right':
        new_orientation = [0.05, -0.09, 0.99, 0.03]
        #raise NotImplementedError('no gripper on right arm')
    elif grasp_orientation == "vertical" and side == 'right':
        new_orientation = [0.05, -0.09, 0.99, 0.03]
        #raise NotImplementedError('no gripper on right arm')
    else:
        new_orientation = [0.004814, -0.02624, -0.99963, 0.0049]
    
    
    pre_grasp_pose = get_pre_grasp_pose(object_position)
    move_pre_grasp_result = move_single_arm(side,pre_grasp_pose['end_effector_position'], new_orientation)
    grasp_pose = get_grasp_pose(object_position)
    move_grasp_result = move_single_arm(side,grasp_pose['end_effector_position'], new_orientation)
    use_gripper_result = use_gripper(side,True)
    return {'move_pre_grasp_result':move_pre_grasp_result,
            'move_grasp_result':move_grasp_result,
            'use_gripper_result':use_gripper_result}

class MoveSingleArmInput(BaseModel):
    """Inputs for move_single_arm"""
    side: str = Field(description="Which robot arm to use, the left or the right", examples=['left','right'])
    coordinates: list[float] = Field(description="The coordinates of the end effector of the robot in meters", examples=[[0.423,0.123,0.234],[-0,324,0.533,0.543]])
    orientation: list[float] = Field(description="The rotation of the end effector on the manipulator represented in quaturnions [x, y, z, w]", examples=[[1.0,0.0,1.0,0.0]]) #prehabs add more examples 

class MoveSingleArmTool(BaseTool):
    name = "move_single_arm"
    description = """
        Useful for when you want to move a single arms end effector to a set of coordinates and a given orientation. 
        When using this tool use the current orientation of the end effecter.
        If used after grasp_object you should first get_full_robot_pose and use that orientation of the end effector.
        Do not use this tool to move the arm into a grasping position. Just use the grasp_object tool for this.
        Outputs a list where the first value is whether or not the move was a success and the second value is the error message"""
    args_schema: Type[BaseModel] = MoveSingleArmInput
    def _run(self, side, coordinates, orientation):
        result = move_single_arm(side, coordinates, orientation)
        return result

class MoveBothArmsInput(BaseModel):
    """Inputs for move_both_arms"""
    left_coordinates: list[float] = Field(description="The coordinates of the left arm end effector in [x ,y, z] order. Must be floats", examples=[[-0.423,0.123,0.234],[-0,324,0.533,0.543]])
    left_orientation: list[float] =  Field(description="The rotation of the left end effector on the manipulator represented in quaturnions [x, y, z, w]", examples=[[0,0,0,1]]) #prehabs add more examples 
    right_coordinates: list[float] = Field(description="The coordinates of the right arm end effector in [x ,y, z] order. Must be floats", examples=[[0.423,0.123,0.234],[0,324,0.533,0.543]])
    right_orientation: list[float] = Field(description="The rotation of the right end effector on the manipulator represented in quaturnions [x, y, z, w]", examples=[[1.0,0.0,1.0,0.0]]) #prehabs add more examples 

class MoveBothArmsTool(BaseTool):
    name = "move_both_arms"
    description = """
        Useful for when you want to move both arms to each their coordinates and end effector orientation. Note that both arms must not be at the same position.
        Outputs a list where the first value is whether or not the move was a success and the second value is the error message"""
    args_schema: Type[BaseModel] = MoveBothArmsInput
    def _run(self,left_coordinates, left_orientation, right_coordinates, right_orientation):
        result = move_both_arms(left_coordinates, left_orientation, right_coordinates, right_orientation)
        return result
    
class UseGripperInput(BaseModel):
    """Inputs for use_gripper"""
    side:str = Field(description="Which arm the gripper is mounted on that you want to use", examples=['left', 'right'])
    state:bool = Field(description="If the gripper should be open or closed eg.: True = close, False = open", examples=[True,False])

class UseGripperTool(BaseTool):
    name = "use_gripper"
    description = """
        Useful for when you want to open or close a gripper.
        Outputs a list where the first value is whether or not the move was a success and the second value is the error message"""
    args_schema: Type[BaseModel] = UseGripperInput
    def _run(self,side,state):
        result = use_gripper(side,state)
        return result

class GetItemDictInput(BaseModel):
    """input for get_item_dict"""
    empty:bool = Field(description="empty value")

class GetItemDictTool(BaseTool):
    name = "get_item_dict"
    description = """
        Usefull for when you want to know what items are pressent in your workspace and where these items are located.
        Outputs a dictionary of the objects presssent and their location"""
    args_schema: Type[BaseModel] = GetItemDictInput
    def _run(self, empty):
        result = get_item_dict(empty)
        return result

class GetFullRobotPoseInput(BaseModel):
    """input for get_full_robot_pose"""
    empty:bool = Field(description="empty value")

class GetFullRobotPoseTool(BaseTool):
    name = "get_full_robot_pose"
    description = """
        Useful for when you want to know the current pose of the robot.
        Outputs a dictionary containing: the left arm's end effector position in the global coordinate system ['left_ee_coor'], 
        the right arm's end effector position in the global coordinate system ['right_ee_coor'], 
        the left arm's end effector orientation in x, y, z, w quaternions ['left_ee_orientation'], 
        the right arm's end effector orientation in quaternions ['right_ee_orientation'], 
        the left arm's gripper state where False = open anf True = closed ['left_gripper_state'], and the right arm's gripper state ['right_gripper_state'] """
    args_schema: Type[BaseModel] = GetFullRobotPoseInput
    def _run(self,empty):
        result = get_full_robot_pose(empty)
        return result
    
class GetPreGraspPoseInput(BaseModel):
    """input for get_pre_grasp_pose"""
    object_position:list[float] = Field(description="The global position on the object that you want to grasp")

class GetPreGraspPoseTool(BaseTool):
    name = "get_pre_grasp_pose"
    description = """
        Useful for when you want to find the end effector pose before going to the grasp postition an object.
        Always use this function before grasping an object by moving the arm that you want to use for grasping to the pose outputted by this function to ensure the correct aproach.
        Outputs a dictionary containing the end effecor position ['end_effector_positiom'] and end effector orientation ['end_effector_orientation']
        The end effector position is in the global frame
        Make sure to always use the get_grasp_pose and go to that position after using this tool before grasping the object"""
    args_schema: Type[BaseModel] = GetPreGraspPoseInput
    def _run(self, object_position):
        pre_grasp_pose = get_pre_grasp_pose(object_position)
        return pre_grasp_pose

class GetGraspPoseInput(BaseModel):
    """input for get_grasp_pose"""
    object_position:list[float] = Field(description="the global position of the object that you want to grasp")

class GetGraspPoseTool(BaseTool):
    name = "get_grasp_pose"
    description = """
        Useful for when you want to find the pose of the end effector to grasp an object 
        Use this after you have moved the arm to the pre grasp pose. Then move the robot to the pose outputted by this tool
        Outputs a dictionary containing the end effecor position ['end_effector_positiom'] and end effector orientation ['end_effector_orientation']
        The end effector position is in the global frame"""
    args_schema: Type[BaseModel] = GetGraspPoseInput
    def _run(self, object_position):
        grasp_pose = get_grasp_pose(object_position)
        return grasp_pose

class GraspObjectInput(BaseModel):
    """input to grasp_object"""
    side:str = Field(description="The robot arm that you want to use to grasp the object",examples=['left','right'])
    object_position: list[float] = Field(description="The global position of the object that you want to grasp")
    grasp_orientation: str = Field(description="The orientation that you want to grasp the object with. Can be 'vertical' or 'horizontal'. It is recommended that you use 'horizontal' for oblong objects such as a bottle and use 'vertical' for smaller objects", examples=['horizontal','vertical'] )

class GraspObjectTool(BaseTool):
    name = "grasp_object"
    description = """
        Useful for when you want to grasp or pick up an object using a single robot arm. When using this tool do not change the position of the arm beforehand. As standard use the horizantal grasp unless it is a small object, eg. : the z position < 0.05.
        This tool executes the following steps: 1. opens the gripper, 2. moves the robot arm to the pre grasp postion, 3. moves the robot arm to the graping position, and 5. closes the gripper around the object.
        Outputs a dictionary containing the response data from each step."""
    args_schema: Type[BaseModel] = GraspObjectInput
    def _run(self,side,object_position, grasp_orientation):
        return grasp_object(side=side,object_position=object_position, grasp_orientation=grasp_orientation)

class LLMPlanningInput(BaseModel):
    """input to use_llm_to_plan"""
    task:str = Field(description="The task provided by the user")





class LLMPlanningTool(BaseTool):
    name = 'use_llm_to_plan'
    description = "Useful for when the task needs to be solved using multiple steps. \nOutputs a more detaild plan to follow"
    args_schema: Type[BaseModel] = LLMPlanningInput
    def _run(self, task):
        items = get_item_dict()
        system_message_planner = """You are a helpful assistant that creates a detailed plan but with as few steps as possible for a robot to follow to solve the task given by the user.
To solve the task, tools from this list can be used:
Use "move_single_arm" to move a single arm's end effector to specified coordinates and orientation. Do not use this function before grasping an object. The grasp_object function handles this part of the grasping
If no orientation is specified use the current orientation. 
Use "move_both_arms" to move both arms to their respective coordinates and end effector orientations (ensuring they are not at the same position).
Use "use_gripper" to open or close the gripper as needed.
Utilize "get_item_dict" to identify objects in the workspace and their locations.
Use "get_full_robot_pose" to retrieve the current pose of the robot, including arm positions, orientations, and gripper states.
Use "grasp_object" to grasp an object with a single robot arm. This tool does all the grasping for you.
Always refer to the grasp_object when grasping or picking up and object and ensure that the other arm than the grasping one is out of the way.
"""
    
        user_template = """Create a plan to {task} using only 200 characters. The items avilable in the the workspace are [{items}]"""

        #defining the prompt template 
        chat_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_message_planner),
                ("human", user_template),
            ]
        )
        model = ChatOpenAI(
            temperature=0,
            model_name="gpt-4-1106-preview"
        )
        msg = chat_template.format_messages(task = task, items = items)

        return model(msg)



#------------------------end of llm tools-----------------------------





#-----------------------robot planner comunication node---------------
class GripperClient(Node):
    def __init__(self):
        super().__init__('gripper_client')
        self.cli = self.create_client(Robotiq3FGripperOutputService, 'Robotiq3FGripper/OutputRegistersService')
        while not self.cli.wait_for_service(timeout_sec = 1.0):
            self.get_logger().info('gripper service not available, waiting again...')

    def send_request(self, state:bool):
        #creating empty request 
        self.req = Robotiq3FGripperOutputService.Request()
        self.req.output_registers.r_act = 1 #activation morty says
        self.req.output_registers.r_mod = 1 #girpper mode 1 = pinch, 0 = basic 
        self.req.output_registers.r_gto = 1 #must be 1 
        self.req.output_registers.r_atr = 0 # must be 0 
        self.req.output_registers.r_spa = 128 #speed must be between 0 and 255
        self.req.output_registers.r_fra = 10 #force must be between 0 and 255

        if state:
            self.req.output_registers.r_pra = 255 #position 0 = open 255 closed 
        else:
            self.req.output_registers.r_pra = 1

        self.future = self.cli.call_async(self.req)

        #self.get_logger().info(f"node exec:{self.executor}, future exec:{self.future._executor}")

        rclpy.spin_until_future_complete(node=self, future=self.future)
        
        #self.get_logger().info("result recived")
        return self.future.result()   





class PlannerClient(Node):
    def __init__(self):
        super().__init__('planner_client')
        self.cli = self.create_client(BimanualJson, 'llm_executor')
        
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
            

    def send_request(self, task):
        #creating empty request 
        self.req = BimanualJson.Request()
        self.req.json_steps = task
        
        self.future = self.cli.call_async(self.req)

        #self.get_logger().info(f"node exec:{self.executor}, future exec:{self.future._executor}")

        rclpy.spin_until_future_complete(node=self, future=self.future)
        
        #self.get_logger().info("result recived")
        return self.future.result()

#-----------------------the gpt controller node----------------------

class Subscriber(Node):
    def __init__(self):
        super().__init__('subscriber')
        
        self._item_dict_sub  = self.create_subscription(String, 
                                                        '/yolo/prediction/item_dict', 
                                                        self.item_dict_callback,
                                                        qos_profile_sensor_data)

        
        self._right_robot_position_sub= self.create_subscription(Pose, 
                                                                 '/right_state_pose', 
                                                                 self.right_robot_position_callback,
                                                                 qos_profile_sensor_data)
        
        self._left_robot_position_sub = self.create_subscription(Pose, 
                                                                 '/left_state_pose', 
                                                                 self.left_robot_position_callback,
                                                                 qos_profile_sensor_data)
        
        self._gripper_state_sub = self.create_subscription(Robotiq3FGripperInputRegisters,
                                                           '/Robotiq3FGripper/InputRegisters',
                                                           self.gripper_state_callback,
                                                           qos_profile_sensor_data)
        
    def item_dict_callback(self, msg):
        #self.get_logger().info(f"item dict recived with length {len(msg.data)}")
        self.item_dict = msg.data
        global item_dict_global
        item_dict_global = json.loads(self.item_dict)
        #self.get_logger().info(f"new global item dict {item_dict_global}")

    #we need to add gripper state callback 
    def right_robot_position_callback(self, msg):
        global robot_full_pose
        robot_full_pose['step_1']['right_ee_coor']=[msg.position.x,
                                                    msg.position.y,
                                                    msg.position.z]
        
        robot_full_pose['step_1']['right_ee_orientation'] = [msg.orientation.x,
                                                             msg.orientation.y,
                                                             msg.orientation.z,
                                                             msg.orientation.w]
        #self.get_logger().info(f"right robot pose recieved, new ee pos {robot_full_pose['step_1']['right_ee_coor']}")


    def left_robot_position_callback(self, msg):
        global robot_full_pose
        robot_full_pose['step_1']['left_ee_coor']= [msg.position.x,
                                                    msg.position.y,
                                                    msg.position.z]
        
        robot_full_pose['step_1']['left_ee_orientation'] = [msg.orientation.x,
                                                             msg.orientation.y,
                                                             msg.orientation.z,
                                                             msg.orientation.w]
        #self.get_logger().info(f"left robot pose recieved, new ee pos {robot_full_pose['step_1']['left_ee_coor']}")
        
    
    def gripper_state_callback(self, msg):
        #self.get_logger().info(f"{msg.g_pra}")
        global robot_full_pose
        gripper_pos = msg.g_pra
        
        if gripper_pos > 20: robot_full_pose['step_1']['left_gripper_state'] = True
        else: robot_full_pose['step_1']['left_gripper_state'] = False
        #self.get_logger().warn(f"new gripper state {robot_full_pose['step_1']['left_gripper_state']}")
        


class GptController(Node):

    def __init__(self):
        # Here you have the class constructor
        # call the class constructor
        super().__init__('gpt_controlle')
        
        #-------------------communication with other nodes------------------------
        self.srv = self.create_service(UserInput, 
                                       'user_input_srv',
                                       self.user_input_callback) 

        
        #---------------langchain and openai setup---------------------

        #creating the system message for the agent llm 
        system_message_agent = SystemMessage(content="You are controlling a bimanual robot. Use the tools provided to sovle the users problem or task. To solve the users problem start by breaking the task into smaller steps that you can solve using a single tool call for each step")

        #defining the model to ofe with the llm
        #gpt-3.5-turbo-1106
        #gpt-4-1106-preview
        llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)

        #defining the list of tools availble for the agent llm to use 
        tools = [MoveSingleArmTool(), 
                 MoveBothArmsTool(), 
                 UseGripperTool(), 
                 GetItemDictTool(), 
                 GetFullRobotPoseTool(), 
                 #GetPreGraspPoseTool(), 
                 #GetGraspPoseTool(),
                 GraspObjectTool(),
                 LLMPlanningTool()]
        #finaly defining the agent llm 

        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        }

        conversational_memory = ConversationBufferWindowMemory(
            memory_key='memory',
            k=5,
            return_messages=True
        )

        mem = AgentTokenBufferMemory(llm= llm)

        self.agent = initialize_agent(tools = tools, 
                                 llm = llm, 
                                 agent=AgentType.OPENAI_FUNCTIONS, 
                                 verbose=True, 
                                 #max_iterations = 5,
                                 #early_stopping_method = 'generate', 
                                 #agent_kwargs=agent_kwargs, 
                                 return_intermediate_steps = True,
                                 memory = mem )
        
        self.runs = 0
    
    def save_llm_output(self):
        memory = self.agent.memory


    #--------------------------send-request-to-pose-commander-and-await-response------------------------
    def send_req_with_planner_node(self, req):
        planner_client = PlannerClient()
        planner_client.get_logger().info("planner client node created")
        result = planner_client.send_request(req)
        planner_client.destroy_node()
        return result

    #----------------------------callback-functions------------------------------
    def user_input_callback(self, request, response):
        #self.get_logger().info("using the callback")
        #self.get_logger().info(f"recieved msg from user: {request.user_input}")
        #self.get_logger().info("invoking chain")
        result = self.agent.__call__({"input":f"{request.user_input}"})
        response.success = True
        response.msg = str(result)
        run_data = self.agent.memory.buffer
        with open("test_data.txt", "a") as f :
            f.write(str(run_data))
            self.runs += 1 
            f.write("\n###\n")
        
        self.agent.memory.clear()

            

        return response 
    

    #-----------------------functions-used-by-LLM--------------------------------
   

    
def main(args=None):
    # initialize the ROS communication
    rclpy.init(args=args)
    # declare the node constructor

    gpt_controller = GptController()
    subscriber = Subscriber()
    executor = MultiThreadedExecutor() 
    executor.add_node(gpt_controller)
    executor.add_node(subscriber)
    executor.spin()
    #rclpy.spin(gpt_controller)


    # Explicity destroy the node
    gpt_controller.destroy_node()
    # shutdown the ROS communication
    rclpy.shutdown()


if __name__ == '__main__':
    main()
