# data generation command instruction for multiple objects

## Ground truth generation:
- Manipulation with ***one arm***, put pnject 1 and object 2 ***inside*** object 3
  >`python move_and_find_data_gen_by_rule_2_objects_inside.py`
- Manipulation with ***one arm***, put pnject 1 and object 2 ***on*** object 3
  >`python move_and_find_data_gen_by_rule_2_objects_on.py`
- Manipulation with ***two arm***, put pnject 1 and object 2 ***inside*** object 3
  >`python move_and_find_data_gen_by_rule_2_objects_inside_both.py`
- Manipulation with ***two arm***, put pnject 1 and object 2 ***on*** object 3
  >`python move_and_find_data_gen_by_rule_2_objects_on_both.py`
- Manipulation with for tasks with temporal constraints
  >`python move_and_find_data_gen_by_rule_2_objects_temp.py`

For the above script, the following variable needs to be defined:
- **-mode**: 'train' or 'test' to set up the running environment.
- **-skip**: the environment to be skipped in string format. e.g. '[2,3]'
- **-path**: the directory to save the generated data.
- **-save_task_folder**: the directory where the tasks were saved.
- **-save_task_file**: the filename for specific saved task.
- **-start_from**: the starting task id number. All tasks with id smaller than this would be skipped.
- **-port_num**: port number to connect to Virtualhome interface

## Evaluation data generation with GPT
- Manipulation with ***one arm***, put pnject 1 and object 2 ***inside*** object 3
  >`python inference_exploration_inside_step.py`
- Manipulation with ***one arm***, put pnject 1 and object 2 ***on*** object 3
  >`python inference_exploration_inside_both.py`
- Manipulation with ***two arm***, put pnject 1 and object 2 ***inside*** object 3
  >`python inference_exploration_on_step.py`
- Manipulation with ***two arm***, put pnject 1 and object 2 ***on*** object 3
  >`python inference_exploration_on_both.py`
- Manipulation with for tasks with temporal constraints
  >`python inference_exploration_temp.py`

For the above script, the following variable needs to be defined:
- **-mode**: 'train' or 'test' to set up the running environment.
- **-skip**: the environment to be skipped in string format. e.g. '[2,3]'
- **-path**: the directory to save the generated data.
- **-save_task_folder**: the directory where the tasks were saved.
- **-start_from**: the starting task id number. All tasks with id smaller than this would be skipped.
- **-port_num**: port number to connect to Virtualhome interface
- **-api_key**: API Key to use OPENAI API