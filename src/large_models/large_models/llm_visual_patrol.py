#!/usr/bin/env python3
# encoding: utf-8
# @Author: Aiden
# @Date: 2025/02/20

import re
import time
import math
import rclpy
import ast
import threading
from speech import speech
from rclpy.node import Node
from large_models.config import *
from geometry_msgs.msg import Twist

from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger, SetBool, Empty
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from large_models_msgs.srv import SetModel,SetString
from ros_robot_controller_msgs.msg import RGBState,RGBStates, SetPWMServoState, PWMServoState

if os.environ["ASR_LANGUAGE"] == 'Chinese':
    PROMPT = '''
# 角色
你是一款智能视觉麦轮小车，需要根据输入的内容，生成对应的json指令。

##要求与限制
1.根据输入的动作内容，在动作函数库中找到对应的指令，并输出对应的指令。
2.为动作序列编织一句精炼（10至30字）、风趣且变化无穷的反馈信息，让交流过程妙趣横生。
3.直接输出json结果，不要分析，不要输出多余内容。
4.格式：{'action':['xx', “xx”], 'response':'xx'}

##结构要求##：
- `"action"`键下承载一个按执行顺序排列的函数名称字符串数组，当找不到对应动作函数时action输出[]。 
- `"response"`键则配以精心构思的简短回复，完美贴合上述字数与风格要求。 




输入:寻黑色的线
输出:{"action":['line_following('black')"],"response":"沿着黑线前行，一路无阻!"}

### 任务示例：
输入:沿着黑色的线前行。。
输出:{"action":['line_following('black')"],"response":"沿着黑线前行，一路无阻!"}

输入:沿着脚下的红线，勇往直前。
输出:{"action":[“line_following('red')”], "response":"红线追踪模式启动，跟着红毯走星光大道"}
'''
else:
    PROMPT = '''
# Role
You are an intelligent vision Mecanum wheel car that needs to generate corresponding JSON instructions based on input commands.

## Requirements and Constraints
1. Based on the input action description, find the corresponding command in the action function library and output the matching instruction.
2. Craft a concise (10-30 words), witty and varied feedback message for each action sequence to make the interaction more engaging.
3. Output only the JSON result directly - no analysis or extra content.
4. Format: {'action':['xx', "xx"], 'response':'xx'}

## Structure Requirements:
- The `"action"` key contains an array of function name strings in execution order. Output [] when no matching action is found.
- The `"response"` key contains a carefully crafted short reply that perfectly matches the word count and style requirements.


### Task Examples:
Input: Follow the black line forward...
Output: {"action":["line_following('black')"],"response":"Tracking black line, smooth sailing ahead!"}

Input: Follow the red line at your feet, march forward bravely.
Output: {"action":["line_following('red')"], "response":"Red line tracking activated, walking the red carpet like a star!"}
'''

class FunctionCall(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name)
        
        
        self.action = []
        self.interrupt = False
        self.llm_result = ''
        self.running = True
        self.result = ''
        
        self.pattern_tracking = r"object_tracking\(['\"]?(.*?)['\"]?\)"
        self.pattern_following = r"line_following\(['\"]?(.*?)['\"]?\)"

        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.sonar_rgb_pub = self.create_publisher(RGBStates, 'sonar_controller/set_rgb', 10)   
        self.rgb_pub = self.create_publisher(RGBStates, 'ros_robot_controller/set_rgb', 10)    
        
        timer_cb_group = ReentrantCallbackGroup()
        self.tts_text_pub = self.create_publisher(String, 'tts_node/tts_text', 1)           
        self.awake_client = self.create_client(SetBool, 'vocal_detect/enable_wakeup')
        self.create_subscription(String, 'agent_process/result', self.llm_result_callback, 1)      
        self.create_subscription(Bool, 'tts_node/play_finish', self.play_audio_finish_callback, 1, callback_group=timer_cb_group)
        
        
        self.awake_client = self.create_client(SetBool, 'vocal_detect/enable_wakeup')
        self.awake_client.wait_for_service()
        self.create_subscription(Bool, 'vocal_detect/wakeup', self.wakeup_callback, 1)
        
        self.set_model_client = self.create_client(SetModel, 'agent_process/set_model')
        self.set_model_client.wait_for_service()

        self.set_prompt_client = self.create_client(SetString, 'agent_process/set_prompt')
        self.set_prompt_client.wait_for_service()
        # line_following client
        self.enter_client_line_following = self.create_client(Trigger, 'line_following/enter')
        self.enter_client_line_following.wait_for_service()
        

        self.start_client_line_following = self.create_client(SetBool, 'line_following/set_running')
        self.start_client_line_following.wait_for_service()

        self.set_target_client_line_following = self.create_client(SetString, 'line_following/set_large_model_target_color')
        self.set_target_client_line_following.wait_for_service()
                        
        msg = SetModel.Request()
        msg.model = llm_model
        msg.model_type = 'llm'
        msg.api_key = api_key 
        msg.base_url = base_url
        
        self.set_model_client.call_async(msg)
        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

    def init_process(self):
        self.timer.cancel()
        msg = SetString.Request()
        msg.data = PROMPT
        self.set_prompt_client.call_async(msg)
        #self.send_request(self.set_prompt_client, msg)
        speech.play_audio(start_audio_path)  
        self.mecanum_pub.publish(Twist())     

        threading.Thread(target=self.process, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)       
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')  
           

    def get_node_state(self, request, response):
        response.success = True
        return response

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()
    
    def wakeup_callback(self, msg):
        if msg.data and self.llm_result:
            self.interrupt = True
    
   
    def llm_result_callback(self, msg):
        self.get_logger().info(msg.data)       
        self.llm_result = msg.data 
        self.get_logger().info('msg:{}'.format(msg.data))
        
    def play_audio_finish_callback(self, msg):
        if msg.data:
            msg = SetBool.Request()
            msg.data = True
            #self.send_request(self.awake_client, msg)
            self.awake_client.call_async(msg)
    
    def process(self):
        while self.running:           
            if self.llm_result != '':
                if 'action' in self.llm_result:
                    self.result = eval(self.llm_result[self.llm_result.find('{'):self.llm_result.find('}')+1])
                    #dict_str = self.llm_result[self.llm_result.find('{'):self.llm_result.find('}') + 1]
                    #self.result = ast.literal_eval(self.result)
                    if 'action' in self.result:
                        action_list = self.result['action']
                    if 'response' in self.result:
                        response = self.result['response']
                    else:
                        response = self.result
                else:
                    time.sleep(0.02)
                response_msg = String()
                response_msg.data = response
                self.tts_text_pub.publish(response_msg)

                
                for a in action_list: 
                    if self.interrupt:
                            self.get_logger().info('interrupt')
                            break                                                                     
                    #elif a.startswith("object_tracking"):
                    if re.search(self.pattern_tracking, a):
                        #color = a.split('"')[1]  
                        color = re.search(self.pattern_tracking, a).group(1)                                                
                        self.send_request(self.enter_client_object_tracking, Trigger.Request())                                          
                        msg = SetString.Request()
                        msg.data = color
                        self.send_request(self.set_target_client_object_tracking, msg)
                        
                        msg = SetBool.Request()
                        msg.data = True
                        self.send_request(self.start_client_object_tracking, msg)                       
                    #elif a.startswith("line_following("):
                    elif re.search(self.pattern_following, a):                    
                        #color = a.split('"')[1]     
                        color = re.search(self.pattern_following, a).group(1)                   
                        self.send_request(self.enter_client_line_following, Trigger.Request())
                        
                        msg = SetString.Request()
                        msg.data = color
                        self.send_request(self.set_target_client_line_following, msg)
                        
                        msg = SetBool.Request()
                        msg.data = True
                        self.send_request(self.start_client_line_following, msg)                
                    else: 
                        self.get_logger().warn("{} is not find".format(a))

                
                self.interrupt = False
                # After executing the action, wait for the next instruction 
                self.llm_result = ''
                msg = SetBool.Request()
                msg.data = True
                self.send_request(self.awake_client, msg)
            else: 
                time.sleep(0.01)
                

        
        
def main():
    node = FunctionCall('function_call')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('shutdown')
    finally:
        rclpy.shutdown() 

if __name__ == "__main__":
    main()
