extends Node2D

var af = Activation.new()
# Called when the node enters the scene tree for the first time.
func _ready():
	#af.init_functions()
	print(af.get_functions().SIGMOID)
# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
