extends Node2D

var saved_files = []

func _ready():
	var file_dialog = $FileDialog
	#file_dialog.mode = FileDialog.FILE_MODE_OPEN_FILES
	#file_dialog.access = FileDialog.ACCESS_FILESYSTEM
	file_dialog.filters = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.mp4", "*.avi", "*.mkv"]
	load_file_metadata()
	# Initialize your UI here based on 'saved_files'

func _on_FilesSelected(paths):
	print(paths)
	for path in paths:
		var filename = path.get_file()
		var user_path = "user://" + filename
		var error = copy_file_to_user(path, user_path)
		if error == OK:
			saved_files.append(user_path)
	save_file_metadata()
	# Update your UI here based on 'saved_files'

func copy_file_to_user(src_path, dest_path):
	var src_file = FileAccess.open(src_path, FileAccess.READ)
	var content = src_file.get_buffer(src_file.get_len())
	src_file.close()

	var dest_file = FileAccess.open(dest_path, FileAccess.WRITE)
	dest_file.store_buffer(content)
	dest_file.close()
	return FAILED

func save_file_metadata():
	var file = FileAccess.open("user://saved_files.dat", FileAccess.WRITE)
	file.store_var(saved_files)
	file.close()

func load_file_metadata():
	if FileAccess.file_exists("user://saved_files.dat"):
		var file = FileAccess.open("user://saved_files.dat", FileAccess.READ)
		saved_files = file.get_var()
		file.close()

# Implement UI functions to display images and videos from 'saved_files'
# Remember to handle different file types appropriately


func _on_file_dialog_file_selected(path):
	print(path)


func _on_file_dialog_dir_selected(dir):
	print(dir)
