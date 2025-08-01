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


task_template = ["""Please find {space1}""",\
                """Kindly locate {space1}""", \
                """Could you retrieve {space1}?""", \
                """ Your task is to get {space1}"""]


for i in tqdm(range(parser.envs)):
    idx = 0
    os.makedirs(parser.save_task_file, exist_ok=True)
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
                "Lamps",
                "Props",
                "Decor",
                "Electronics",
                "Food",
            ]
        )
    object_names = list(set([obj["class_name"] for obj in object_list]))
    with open(f"{parser.save_task_file}/env_{i}.json", "a", encoding="utf-8") as f:
        for obj in tqdm(object_names):
            if obj not in obj_info:
                continue
            selected = sample(obj_info[obj], 1)
            e1, e2, rel, id1, id2 = selected[0]
            space = f"{e1} {rel.lower()} {e2}"
            template = sample(task_template, 1)[0]
            task = template.format(space1=e1)
            completion_criterion = f"(CLOSE, character, {obj})({rel}, {e1}, {e2})"
            new = {
                "task": task,
                "task_completion_criterion": completion_criterion,
                "object_1": obj,
                "object_2": e2,
                "env_id": i,
                "instant_id_1": id1,
                "instant_id_2": id2,
                "index": idx
            }
            f.write(json.dumps(new, ensure_ascii=False) + "\n")
            idx += 1
