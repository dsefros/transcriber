import os

def collect_python_files(root_dir: str = ".", output_file: str = "collected_code.txt", exclude_dirs: set = None):
    if exclude_dirs is None:
        exclude_dirs = {"venv", "__pycache__"}

    with open(output_file, "w", encoding="utf-8") as out_f:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Пропускаем исключённые директории
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
            for filename in sorted(filenames):
                if filename.endswith(".py"):
                    filepath = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(filepath, root_dir)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                        out_f.write(f"***\n{rel_path}\n***\n{content}\n***\n")
                    except Exception as e:
                        out_f.write(f"***\n{rel_path}\n***\n<Ошибка при чтении файла: {e}>\n***\n")

if __name__ == "__main__":
    collect_python_files()