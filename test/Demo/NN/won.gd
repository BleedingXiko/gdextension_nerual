extends Area2D

func _on_won_body_entered(body: Node):
#	print(body)
	if body.is_in_group("ai"):
		body.time_alive -= body.time_alive / 2
		body.queue_free()
		print("AI Won")


func _on_neural_net_gen_changed(_generation):
	var maps = ["res://Demo/NN/track.tscn", "res://Demo/NN/track_2.tscn"]
	var picked = maps.pick_random()
	var scene = load(picked)
	var child = scene.instantiate()
	add_child(child)
	$"../gen".text = str(_generation)


func _on_neural_net_true_batch_size(_size):
	print("True Batch Size: ", _size)

