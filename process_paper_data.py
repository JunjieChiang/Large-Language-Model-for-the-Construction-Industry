from src.single_choice.together import work as stw
import os
def process(process_type, paper_path):

    file_name, file_extension = os.path.splitext(os.path.basename(paper_path))

    if process_type == 0:
        stw.main(paper_path, file_name)