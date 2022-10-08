'''
reference
----------
https://github.com/kakaoenterprise/JORLDY/blob/master/jorldy/core/agent/__init__.py
'''

import os
import inspect


working_dir = os.path.dirname(os.path.realpath(__file__))
file_list = os.listdir(working_dir)
module_list = [
    file.replace('.py', '')
    for file in file_list
    if file.endswith('.py') and file not in ['__init__.py']
]

agent_dict = {}
for module_name in module_list:
    module_path = f"{__name__}.{module_name}"
    module = __import__(module_path, fromlist=[None])
    for class_name, _class in inspect.getmembers(module, inspect.isclass):
        if module_path in str(_class):
            agent_dict[class_name] = _class


class Agent:
    def __new__(self, name, *args, **kwargs):
        return agent_dict[name](*args, **kwargs)
