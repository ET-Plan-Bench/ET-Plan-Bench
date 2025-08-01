# Task generation command instruction

## Ground truth generation:
- Generate simple navigation task
  >`python create_simple.py`
- Generate simple navigation task with spatial relation constraints
  >`python create_simple_cons.py`
- Generate navigation & manipuation task for one object
  >`python create_nav_man.py`
- Generate navigation & manipuation task for one object with spatial relation constraints
  >`python create_nav_man_cons.py`
- Generate navigation & manipuation task for two objects
  >`python create_nav_multi.py`
- Generate navigation & manipuation task for two objects with temporal constraints
  >`python create_temp.py`

For the above script, the following variable needs to be defined:
- **-rel**: 'inside' or 'on' to set up the manipuation type.
- **-envs**: the largest environment to generate tasks for. Integer.
- **-path**: the directory to save the generated data.
- **-save_task_file**: the directory where the tasks are saved.
- **-env_graph_file**: the directory where the scene graph information is saved
- **-api_key**: API Key to use OPENAI API