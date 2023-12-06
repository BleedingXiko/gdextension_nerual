extends Node2D

var af = Activation.new().functions
# Called when the node enters the scene tree for the first time.
func _ready():
	seed(0)
	var m: Matrix = Matrix.new()
	m.init(8,6)
	m = Matrix.rand(m)
	var t: Matrix = Matrix.from_array([2, randf(), randf(),randf()])
	var x: Array = Matrix.to_array(t)
	
	var temp = Matrix.map(m, af.SWISH.function)
	var s = temp.save()
	var l = Matrix.load(s)
	print(temp.get_data())


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
