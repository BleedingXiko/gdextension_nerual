extends Node2D

var qt: QTable
var qnet: QNetwork
var row: int = 0
var column: int = 0

var reward_states = [4, 24, 35]
var target: int = reward_states.pick_random()
var punish_states = [3, 18, 21, 28, 31]

var current_state: Array = []
var previous_reward: float = 0.0

var total_iteration_rewards: Array[float] = []
var current_iteration_rewards: float = 0.0
var done: bool = false


var ACTIVATIONS = Activation.new().functions


var q_network_config = {
	"print_debug_info": true,
	"exploration_probability": 1.0,
	"exploration_decreasing_decay": 0.01,
	"min_exploration_probability": 0.05,
	"discounted_factor": 0.95,
	"decay_per_steps": 250,
	"use_replay": true,
	"is_learning": true,
	"use_target_network": true,
	"update_target_every_steps": 1500,
	"memory_capacity": 800,
	"batch_size": 256, 
	#used by the neural network
	"learning_rate": 0.0000001, 
	"l2_regularization_strength": 0.001,
	"use_l2_regularization": false,
}
var q_table_config = {
	"print_debug_info": false,
	"exploration_decreasing_decay": 0.01,
	"min_exploration_probability": 0.02,
	"discounted_factor": 0.9,
	"learning_rate": 0.1,
	"decay_per_steps": 100,
	"max_state_value": 2,
	"random_weights": false,
}

func _ready() -> void:
	qt = QTable.new(36 * 3, 4, q_table_config)
	qnet = QNetwork.new(q_network_config) #config is used by both qnet and neural network advanced
	qnet.add_layer(2) #input nodes
	qnet.add_layer(8, ACTIVATIONS.TANH) #hidden layer
	qnet.add_layer(4, ACTIVATIONS.SIGMOID) # 4 actions

func _process(_delta: float) -> void:
	if Input.is_action_just_pressed("predict"):
		$Timer.wait_time = 0.5
	elif Input.is_action_just_pressed("ui_down"):
		$Timer.wait_time = 0.001
	elif Input.is_action_just_pressed("ui_up"):
		qnet.save('./qnet.data')

func _on_timer_timeout():
	if done:
		reset()
		return
	current_state = [row * 6 + column, target]
	var action_to_do: int = qnet.predict(current_state, previous_reward)
	
	current_iteration_rewards += previous_reward
	previous_reward = 0.0
		
	if is_out_bound(action_to_do):
		previous_reward -= 0.75
		done = true
	elif row * 6 + column in punish_states:
		previous_reward -= 0.5
		done = true
	elif (row * 6 + column) == target:
		previous_reward += 1.0
		done = false
		target = reward_states.pick_random()
	else:
		previous_reward -= 0.05
	$player.position = Vector2(96 * column + 16, 512 - (96 * row + 16))
	$lr.text = str(qnet.exploration_probability)
	$target.text = str(target)


func is_out_bound(action: int) -> bool:
	var _column := column
	var _row := row
	match action:
		0:
			_column -= 1
		1:
			_row += 1
		2:
			_column += 1
		3:
			_row -= 1
	if _column < 0 or _row < 0 or _column > 5 or _row > 5:
		return true
	else:
		column = _column
		row = _row
		return false

func reset():
	target = reward_states.pick_random()
	row = randi_range(1, 5)
	column = randi_range(1, 5)
	done = false
	total_iteration_rewards.append(current_iteration_rewards)
	current_iteration_rewards = 0.0
	$player.position = Vector2(96 * column + 16, 512 - (96 * row + 16))

