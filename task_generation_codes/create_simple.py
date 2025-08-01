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


os.makedirs(parser.save_task_file, exist_ok=True)
for i in tqdm(range(parser.envs)):
    idx = 0
    if i == 9:
        continue
    graph_file = f"{parser.env_graph_file}/graph_{i}.json"
    graph = load_json(graph_file)
    scene_graph = VirtualHomeSceneGraph(graph)

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
            template = sample(task_template, 1)[0]
            task = template.format(space1=obj)
            completion_criterion = f"(CLOSE, character, {obj})"
            new = {
                "task": task,
                "task_completion_criterion": completion_criterion,
                "object_1": obj,
                "env_id": i,
                "index": idx
            }
            f.write(json.dumps(new, ensure_ascii=False) + "\n")
            idx += 1
