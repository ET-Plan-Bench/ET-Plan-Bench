# EmbodiedAI - file structures
## constant: all constant strings
* prompt: all prompts used
    * data_gen.py
* dirs.py: all directories
* llm.py: gpt-related parameters
* params.py: all other parameters
## envs: environment classes
* env.py: virtual home environment and task-specific environment
* scene_graph.py: graph representation of environment information, holded by environments in env.py
## evalutate
* evaluate_nav_mani_occlusion.py
* evaluate_nav_mani.py
## data_gen: task generation
## Log: all logs
## Output: all outputs
## utils: all util functions
* eval.py: evaluation-related functions
* gpt_call.py
* helper.py
* io.py
* log.py
* robot_task.py: task-level robot manipulation
* robot.py: basic components of robot manipulation
* unity.py: connection to virtual home hosted by unity


# Get started
## Info
__task_name_id__:
* 0: navigation and put something on something
* 1: navigation and put something inside something
* 2: navigation and put something on something with constraints
* 3: navigation and put something inside something with constraints
## Data generation
1. Filter invalid task by running virtualhome with ground truth knowledge:
>`python et-plan-bench/evaluation_generation/navigation_manipulation_tasks/EmbodiedAI/data_gen/ground_truth_single_task_type.py --task_name_id 0`
2. Run the tasks in virtualhome to generate data
>`python et-plan-bench/evaluation_generation/navigation_manipulation_tasks/EmbodiedAI/data_gen/visible_single_task_type.py --task_name_id 0`

## Evaluation
* Generate evaluation results for tasks without constraints
>`python et-plan-bench/evaluation_generation/navigation_manipulation_tasks/EmbodiedAI/evaluate/evaluate_nav_mani.py`
* Generate evaluation results for tasks with constraints
>`python et-plan-bench/evaluation_generation/navigation_manipulation_tasks/EmbodiedAI/evaluate/evaluate_nav_mani.py --constraint`
* Generate evaluation results for item-size occlusion
>`python et-plan-bench/evaluation_generation/navigation_manipulation_tasks/EmbodiedAI/evaluate/evaluate_nav_mani_occlusion.py --occlusion item_size`
* Generate evaluation results for initial-distance occlusion
>`python et-plan-bench/evaluation_generation/navigation_manipulation_tasks/EmbodiedAI/evaluate/evaluate_nav_mani_occlusion.py --occlusion init_distance`
