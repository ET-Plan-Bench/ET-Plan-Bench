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


prompt_tasks = """I will give you a task to put object 1 related to object 3 on or inside the object 2 related to object 4. Please make a decision whether it makes sense in the realistic world. Besides, there are rules that needs to be considered. If the task does not meet the rules, it is still considered as unreasonable

1. The object 1 needs to be grabbable reasonably for a human.
2. The object 1 needs to be smaller than the object 2.
3. The task only requires the robot to move, find, grab, open, close, put in or out.
4. Based on common sense, the task needs to happen in realistic world. For instance, putting desk into the frying pan is not a reasonable task. 

In the following, we provide several examples.

Example:
task: Find the tube of toothpaste close to the water glass and place it inside the bathroom cabinet in the bathroom.
conclusion: Reasonable

task: Find the cooking pot inside the kitchen and place it in the sink close to the faucet
conclusion: Reasonable

task: Can you take the shirt from inside the closet and hang it on the towel rack close to the wall?
conclusion: Reasonable

task: Can you move the condiment shaker close to the salad and place it onto the kitchen counter inside the kitchen?
conclusion: Reasonable

task: Locate the keyboard on the desk and place it carefully inside the box in the living room.
conclusion: Unreasonable

task: Locate the cutlery fork close to the condiment bottle and place it on the bookshelf inside the bedroom.
conclusion: Unreasonable

task: Please place the magazine from inside the living room on the cutting board inside the kitchen.
conclusion: Unreasonable

Please only output 'Reasonable' or 'Unreasonable' as output. Do not output thinking process, or extra words such as 'conclusion' or API_key

task: {task}

"""
if parser.rel == "inside":
    task_template = ["""Please find {space1} and put it into {space2}.""",\
                    """Kindly locate {space1} and place it in {space2}.""", \
                    """Could you retrieve {space1} and place it into {space2}?""", \
                    """ Your task is to get {space1} and move it to {space2}."""]
elif parser.rel == "on":
    task_template = ["""Please find {space1} and put it on {space2}.""",\
                    """Kindly locate {space1} and place it on {space2}.""", \
                    """ I need you to search for {space1} and position it on {space2}.""", \
                    """ Your task is to grab {space1} and move it onto {space2}."""]

for i in tqdm(range(parser.envs)):
    if i == 9:
        continue
    graph_file = f"{parser.env_graph_file}/graph_{i}.json"
    graph = load_json(graph_file)
    scene_graph = VirtualHomeSceneGraph(graph)

    nodes = [scene_graph._get_node(i) for i in scene_graph.nodes]
    room_ids = []
    for node in nodes:
        if node["category"] == "Rooms":
            room_ids.append(node["id"])
    room_sep_dict = defaultdict(list)
    room_info = {}
    for edge in scene_graph.edges:
        id1 = scene_graph._get_node(edge[0])["id"]
        id2 = scene_graph._get_node(edge[1])["id"]
        rel = scene_graph.edges[edge]["relation_type"]
        if id1 in room_ids:
            room_sep_dict[id1].append(id2)
            room_info[id2] = id1

        if id2 in room_ids:
            room_sep_dict[id2].append(id1)
            room_info[id1] = id2

    obj_info = defaultdict(list)
    for edge in scene_graph.edges:
        e1 = scene_graph._get_node(edge[0])["class_name"]
        e2 = scene_graph._get_node(edge[1])["class_name"]
        id1 = scene_graph._get_node(edge[0])["id"]
        id2 = scene_graph._get_node(edge[1])["id"]
        cat1 = scene_graph._get_node(edge[0])["category"]
        cat2 = scene_graph._get_node(edge[1])["category"]
        if cat1 == "Rooms" or cat2 == "Rooms":
            rel = scene_graph.edges[edge]["relation_type"]
            obj_info[e1].append([e1, e2, rel, id1, id2]) 
        else:
            room1 = room_info[id1]
            room2 = room_info[id2]
            if room1 == room2 :
                rel = scene_graph.edges[edge]["relation_type"]
                obj_info[e1].append([e1, e2, rel, id1, id2])
    
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
        all_comb = list(itertools.product(object_names_1, object_names_2))
    elif parser.rel == "on":
        object_names_1 = list(set([obj["class_name"] for obj in object_list if 'SURFACES' not in obj['properties'] and 'GRABBABLE' in obj['properties'] and "MOVABLE" in obj['properties']]))
        object_names_2 = list(set([obj["class_name"] for obj in object_list if 'SURFACES' in obj['properties']]))
        all_comb = list(itertools.product(object_names_1, object_names_2))
    
    
    idx = 0
    os.makedirs(parser.save_task_file, exist_ok=True)
    with open(f"{parser.save_task_file}/env_{i}.json", 'a', encoding="utf-8") as f:
        for obj1, obj2 in all_comb:
            if obj1 not in obj_info or obj2 not in obj_info:
                continue

            selected1 = sample(obj_info[obj1], 1)
            e1, e3, rel1, id1, id3 = selected1[0]
            stop_max = 5
            stop_try = 0
            while e3 == obj2 and stop_try<stop_max:
                selected1 = sample(obj_info[obj1], 1)
                e1, e3, rel1, id1, id3 = selected1[0]
                stop_try += 1
            if stop_try == stop_max:
                continue
            stop_try = 0
            selected2 = sample(obj_info[obj2], 1)
            e2, e4, rel2, id2, id4 = selected2[0]
            while e4 == obj1 and stop_try<stop_max:
                selected2 = sample(obj_info[obj2], 1)
                e2, e4, rel2, id2, id4 = selected2[0]
                stop_try += 1
            if stop_try == stop_max:
                continue
            template = sample(task_template, 1)[0]
            space1 = f"{e1} {rel1.lower()} {e3}"
            space2 = f"{e2} {rel2.lower()} {e4}"
            task = template.format(space1=space1, space2=space2)
            response = query_gpt_single(prompt_tasks.format(task=task))
            if not 'Reasonable' in response:
                continue
            if parser.rel == "inside":
                completion_criterion = f"(CLOSE, robot, {obj1})(CLOSE, robot, {obj2})(INSIDE, {obj1}, {obj2})({rel1}, {e1}, {e3})({rel2}, {e2}, {e4})"
            elif parser.rel == "on":
                completion_criterion = f"(CLOSE, robot, {obj1})(CLOSE, robot, {obj2})(ON, {obj1}, {obj2})({rel1}, {e1}, {e3})({rel2}, {e2}, {e4})"                
            new = {
            "task": task,
            "task_completion_criterion": completion_criterion,
            "object1": e1,
            "object2": e2,
            "object3": e3,
            "object4": e4,
            "env_id": i,
            "index": idx,
            "id1": id1,
            "id2": id2,
            "id3": id3,
            "id4": id4,
            "rel1": rel1,
            "rel2": rel2
            }
            f.write(json.dumps(new, ensure_ascii=False) + "\n")
            idx += 1




