from openai import OpenAI
import json
from scene_graph import VirtualHomeSceneGraph
import itertools
from tqdm import tqdm
import random
import time
import sys
from collections import defaultdict
from random import sample
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--api_key", "-api_key", type=str, help="api_key to use OPENAI API")
parser.add_argument("--rel", "-rel", type=str, help="The type of relation defined for the task")
parser.add_argument("--envs", "-envs", type=int, help="The environment number for create task")
parser.add_argument("--env_graph_file", "-env_graph_file", type=str, help="The directory of environment scene graphs")
parser.add_argument("--save_task_file", "-save_task_file", type=str, help="The directory to save generated tasks")
parser = parser.parse_args()

API_key = parser.api_key


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def retry_fn(fn, max_failures=10, sleep_time=30):
    failures = 0
    while failures < max_failures:
        try:
            return fn()
        except KeyboardInterrupt:
            print('Interrupted')
            sys.exit(0)
        except Exception as e:
            failures += 1
            print('Failed with exception:')
            print(e)
            print(f'Failed {failures} times, waiting {sleep_time}s to retry')
            time.sleep(sleep_time)
            if failures >= max_failures:
                raise Exception('Max failures exceeded.')
            time.sleep(2)

def query_gpt_single(prompt, max_tokens=4096, temperature=0.3, user_role=True):

    client=OpenAI(api_key='<KEY>')
    client.api_key=API_key

    def f(prompt, user_role=True):
        if user_role:
            message = [{'role': 'user', 'content': prompt}]
        else:
            message = prompt
        response = client.chat.completions.create(
            model = "gpt-4o",
            messages = message,
            temperature = temperature,
            max_tokens = max_tokens,
        )

        response = response.choices[0].message.content
        return response

    response = retry_fn(lambda: f(prompt, user_role), max_failures=5, sleep_time=5)

    return response


prompt_tasks = """I will give a task to put object 1 and object 2 inside or on object 3. Please make a decision whether it makes sense in the realistic world. Besides, there are rules that needs to be considered. If the task does not meet the rules, it is still considered as unreasonable

1. The object 1 needs to be grabbable reasonably for a human.
2. The object 2 needs to be grabbable reasonable for a human as well
3. The object 1 and object 2 needs to be smaller than the object 3. 
4. The object 1 and object 2 are reasonably related. For instance, a apple and a banana is related as they are both fruits. A apple and a knife are also related, as we can use a knife to cut a apple.
5. Based on common sense, the task needs to make sense in realistic world. For instance, puttting desk and table into the frying pan is not a reasonale task. 

Following are examples:
Example:
task: Find an apple and a knife, then put them into the box
conclusion: Reasonable

task: Can you find knife and the apple, and put them into a tray for later use
conclusion: Reasonable

task: Please put the bowl and the cellphone into the sink
conclusion: Unreasonable

task: Locate the toothbrush and the condiment shaker, then place them on the bed
conclusion: Unreasonable

task: Find a computer mouse and a peach, then place them on the bench
conclusion: Unreasonable

Please only output 'Reasonable' or 'Unreasonable' as output. Do not output thinking process, or extra words such as 'conclusion' and API_key

task: {task}

"""
if parser.rel == "inside":
    task_template = ["""Kindly locate {space1} and {space2}, and place them inside {space3}.""", \
             """I need you to search for {space1} and {space2} and then position them into {space3}.""", \
             """Your task is to gather {space1} and {space2} and subsequently lay them into {space3}.""", \
             """Please find {space1} and {space2}, then put it into {space3}"""]
elif parser.rel == "on":
    task_template = ["""Kindly locate {space1} and {space2}, and place them on {space3}.""", \
             """I need you to search for {space1} and {space2} and then position them onto {space3}.""", \
             """Your task is to gather {space1} and {space2} and subsequently lay them onto {space3}.""", \
             """Please find {space1} and {space2}, then put it on {space3}"""]

for i in tqdm(range(parser.envs)):
    if i == 9:
        continue
    graph_file = f"{parser.env_graph_file}/graph_{i}.json"
    graph = load_json(graph_file)
    scene_graph = VirtualHomeSceneGraph(graph)
    
    object_list = scene_graph.filter_nodes(
    categories=[
        "Furniture",
        "Appliances",
        "Props",
        "Decor",
        "Electronics",
        "Food",
        ]
    )
    if parser.rel == "inside":
        object_names_1 = list(set([obj["class_name"] for obj in object_list if 'CONTAINERS' not in obj['properties'] and 'GRABBABLE' in obj['properties'] and "MOVABLE" in obj['properties']]))
        object_names_2 = list(set([obj["class_name"] for obj in object_list if 'CONTAINERS' in obj['properties']]))
        all_comb_1 = list(itertools.product(object_names_1, object_names_1))
        all_comb = list(itertools.product(all_comb_1, object_names_2))
    elif parser.rel == "on":
        object_names_1 = list(set([obj["class_name"] for obj in object_list if 'SURFACES' not in obj['properties'] and 'GRABBABLE' in obj['properties'] and "MOVABLE" in obj['properties']]))
        object_names_2 = list(set([obj["class_name"] for obj in object_list if 'SURFACES' in obj['properties']]))
        all_comb_1 = list(itertools.product(object_names_1, object_names_1))
        all_comb = list(itertools.product(all_comb_1, object_names_2))
    
    
    idx = 0
    os.makedirs(parser.save_task_file, exist_ok=True)
    with open(f"{parser.save_task_file}/env_{i}.json", 'a', encoding="utf-8") as f:
        for comb in all_comb:
            obj1 = comb[0][0]
            obj2 = comb[0][1]
            obj3 = comb[1]
            if obj1 == obj2:
                continue
            template = sample(task_template, 1)[0]
            task = template.format(space1=obj1, space2=obj2, space3=obj3)
            response = query_gpt_single(prompt_tasks.format(task=task))
            if not 'Reasonable' in response:
                continue
            if parser.rel == "inside":
                completion_criterion = f"(CLOSE, robot, {obj1})(CLOSE, robot, {obj3})(INSIDE, {obj1}, {obj3})(CLOSE, robot, {obj2})(CLOSE, robot, {obj3})(INSIDE, {obj2}, {obj3})"
            elif parser.rel == "on":
                completion_criterion = f"(CLOSE, robot, {obj1})(CLOSE, robot, {obj3})(ON, {obj1}, {obj3})(CLOSE, robot, {obj2})(CLOSE, robot, {obj3})(ON, {obj2}, {obj3})"
            new = {
            "task": task,
            "task_completion_criterion": completion_criterion,
            "object1": obj1,
            "object2": obj2,
            "object3": obj3,
            "env_id": i,
            "index": idx
            }
            f.write(json.dumps(new, ensure_ascii=False) + "\n")
            idx += 1




