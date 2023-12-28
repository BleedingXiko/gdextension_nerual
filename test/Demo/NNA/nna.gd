extends Node2D

var nn: NeuralNetwork

var af = Activation.new()

var activations = af.get_functions()


func _ready() -> void:
	nn = NeuralNetwork.new(3, 4, 1)
	nn.set_activation_function(activations.TANH, activations.SIGMOID)
	
	
	nn.train([1.0, 2.0, 3.0], [6.0])
		#nn.train([4.0, 5.0, 6.0], [15.0])
		#nn.train([7.0, 8.0, 9.0], [24.0])
		#nn.train([10.0, 11.0, 12.0], [33.0])
		#nn.train([13.0, 14.0, 15.0], [42.0])


func _physics_process(delta: float) -> void:
	if Input.is_action_just_pressed("predict"):
		#nnas.save("./test.nn")
		print("--------------Prediction--------------")
		print(nn.predict([1.0, 2.0, 3.0]))
		#print(nn.predict([4.0, 5.0, 6.0]))
		#print(nn.predict([7.0, 8.0, 9.0]))
		#print(nn.predict([10.0, 11.0, 12.0]))
		#print(nn.predict([13.0, 14.0, 15.0]))
		#print(nn.predict([15.0, 15.0, 15.0]))

