class_name NeuralNetwork

var ACTIVATIONS = Activation.new().functions

var best: bool = false
var input_nodes: int
var hidden_nodes: int 
var output_nodes: int

var weights_input_hidden: Matrix
var weights_hidden_output: Matrix

var bias_hidden: Matrix
var bias_output: Matrix

var learning_rate: float = 0.15


var hidden_activation: Dictionary
var output_activation: Dictionary

var fitness: float = 0.0

var color: Color = Color.TRANSPARENT

var raycasts: Array[RayCast2D]

func _init(_input_nodes: int, _hidden_nodes: int, _output_nodes: int, is_set: bool = false) -> void:
	if !is_set:
		randomize()
		input_nodes = _input_nodes;
		hidden_nodes = _hidden_nodes;
		output_nodes = _output_nodes;
		
		var ih = Matrix.new()
		var bh = Matrix.new()
		
		var ho = Matrix.new()
		var bo = Matrix.new()
		
		ih.init(hidden_nodes, input_nodes)
		bh.init(hidden_nodes, 1)
		
		ho.init(output_nodes, hidden_nodes)
		bo.init(output_nodes, 1)
		
		weights_input_hidden = Matrix.rand(ih)
		weights_hidden_output = Matrix.rand(ho)
		
		bias_hidden = Matrix.rand(bh)
		bias_output = Matrix.rand(bo)
	
	set_activation_function()
	set_nn_color()

func set_nn_color():
	color = Color(Matrix.average(weights_input_hidden),
	Matrix.average(weights_hidden_output),
	Matrix.average(bias_hidden))

func set_activation_function(hidden_func: Dictionary = ACTIVATIONS.TANH, output_func: Dictionary = ACTIVATIONS.SIGMOID) -> void:
	hidden_activation = hidden_func
	output_activation = output_func

func predict(input_array: Array) -> Array:
	var inputs = Matrix.from_array(input_array)
	
	var hidden = Matrix.dot_product(weights_input_hidden, inputs)
	hidden = Matrix.add(hidden, bias_hidden)
	hidden = Matrix.map(hidden, hidden_activation.function)

	var output = Matrix.dot_product(weights_hidden_output, hidden)
	output = Matrix.add(output, bias_output)
	output = Matrix.map(output, output_activation.function)

	return Matrix.to_array(output)

func train(input_array: Array, target_array: Array):
	var inputs = Matrix.from_array(input_array)
	var targets = Matrix.from_array(target_array)
	
	var hidden = Matrix.dot_product(weights_input_hidden, inputs);
	hidden = Matrix.add(hidden, bias_hidden)
	hidden = Matrix.map(hidden, hidden_activation.function)
	
	var outputs = Matrix.dot_product(weights_hidden_output, hidden)
	outputs = Matrix.add(outputs, bias_output)
	outputs = Matrix.map(outputs, output_activation.function)
	
	var output_errors = Matrix.subtract(targets, outputs)
	
	var gradients = Matrix.map(outputs, output_activation.derivative)
	gradients = Matrix.multiply(gradients, output_errors)
	gradients = Matrix.scalar(gradients, learning_rate)
	
	var hidden_t = Matrix.transpose(hidden)
	var weight_ho_deltas = Matrix.dot_product(gradients, hidden_t)
	
	weights_hidden_output = Matrix.add(weights_hidden_output, weight_ho_deltas)
	bias_output = Matrix.add(bias_output, gradients)
	
	var weights_hidden_output_t = Matrix.transpose(weights_hidden_output)
	var hidden_errors = Matrix.dot_product(weights_hidden_output_t, output_errors)
	
	var hidden_gradient = Matrix.map(hidden, hidden_activation.derivative)
	hidden_gradient = Matrix.multiply(hidden_gradient, hidden_errors)
	hidden_gradient = Matrix.scalar(hidden_gradient, learning_rate)
	
	var inputs_t = Matrix.transpose(inputs)
	var weight_ih_deltas = Matrix.dot_product(hidden_gradient, inputs_t)

	weights_input_hidden = Matrix.add(weights_input_hidden, weight_ih_deltas)

	bias_hidden = Matrix.add(bias_hidden, hidden_gradient)

func get_inputs_from_raycasts() -> Array:
	assert(raycasts.size() != 0, "Can not get inputs from RayCasts that are not set!")
	
	var _input_array: Array[float]
	
	for ray in raycasts:
		if is_instance_valid(ray): _input_array.push_front(get_distance(ray))
	
	return _input_array

func get_prediction_from_raycasts(optional_val: Array = []) -> Array:
	assert(raycasts.size() != 0, "Can not get inputs from RayCasts that are not set!")
	
	var _array_ = get_inputs_from_raycasts()
	_array_.append_array(optional_val)
	return predict(_array_)

func get_distance(_raycast: RayCast2D):
	var distance: float = 0.0
	if _raycast.is_colliding():
		var origin: Vector2 = _raycast.global_transform.get_origin()
		var collision: Vector2 = _raycast.get_collision_point()
		
		distance = origin.distance_to(collision)
	else:
		distance = sqrt((pow(_raycast.target_position.x, 2) + pow(_raycast.target_position.y, 2)))
	return distance

static func reproduce(a: NeuralNetwork, b: NeuralNetwork) -> NeuralNetwork:
	var result = NeuralNetwork.new(a.input_nodes, a.hidden_nodes, a.output_nodes)
	result.weights_input_hidden = Matrix.random(a.weights_input_hidden, b.weights_input_hidden)
	result.weights_hidden_output = Matrix.random(a.weights_hidden_output, b.weights_hidden_output)
	result.bias_hidden = Matrix.random(a.bias_hidden, b.bias_hidden)
	result.bias_output = Matrix.random(a.bias_output, b.bias_output)
	result.set_activation_function(a.hidden_activation, a.output_activation)

	return result

static func mutate(nn: NeuralNetwork, callback: Callable = Callable(NeuralNetwork, "mutate_callable_reproduced")) -> NeuralNetwork:
	var result = NeuralNetwork.new(nn.input_nodes, nn.hidden_nodes, nn.output_nodes)
	result.weights_input_hidden = Matrix.map(nn.weights_input_hidden, callback)
	result.weights_hidden_output = Matrix.map(nn.weights_hidden_output, callback)
	result.bias_hidden = Matrix.map(nn.bias_hidden, callback)
	result.bias_output = Matrix.map(nn.bias_output, callback)
	result.set_activation_function(nn.hidden_activation, nn.output_activation)
	return result

static func mutate_callable_reproduced(value, _row, _col):
	seed(randi())
	randomize()
	value += randf_range(-0.15, 0.15)
	return value

static func copy(nn : NeuralNetwork) -> NeuralNetwork:
	var result = NeuralNetwork.new(nn.input_nodes, nn.hidden_nodes, nn.output_nodes)
	result.weights_input_hidden = Matrix.copy(nn.weights_input_hidden)
	result.weights_hidden_output = Matrix.copy(nn.weights_hidden_output)
	result.bias_hidden = Matrix.copy(nn.bias_hidden)
	result.bias_output = Matrix.copy(nn.bias_output)
	result.color = nn.color
	result.fitness = nn.fitness
	result.set_activation_function(nn.hidden_activation, nn.output_activation)
	return result

static func mutate_callable(value, _row, _col):
	seed(randi())
	randomize()
	value += randf_range(-0.5, 0.5)
	return value

func save(path):
	var file = FileAccess.open(path, FileAccess.WRITE)
	var network: Dictionary = {
		"input_nodes": input_nodes,
		"hidden_nodes": hidden_nodes,
		"output_nodes": output_nodes,
		"weights_input_hidden": weights_input_hidden.save(),
		"weights_hidden_output": weights_hidden_output.save(),
		"bias_hidden": bias_hidden.save(),
		"bias_output": bias_output.save(),
		"hidden_activation": hidden_activation.name,
		"output_activation": output_activation.name,
	}
	file.store_var(network)
	file.close()


static func load(path) -> NeuralNetwork:
	var file = FileAccess.open(path, FileAccess.READ)
	var data = file.get_var()
	var nn = NeuralNetwork.new(data.input_nodes, data.hidden_nodes, data.output_nodes)

	nn.weights_input_hidden = Matrix.load(data.weights_input_hidden)
	nn.weights_hidden_output = Matrix.load(data.weights_hidden_output)
	nn.bias_hidden = Matrix.load(data.bias_hidden)
	nn.bias_output = Matrix.load(data.bias_output)

	nn.set_activation_function(nn.ACTIVATIONS[data.hidden_activation], nn.ACTIVATIONS[data.output_activation])
	nn.set_nn_color()
	
	file.close()
	return nn
