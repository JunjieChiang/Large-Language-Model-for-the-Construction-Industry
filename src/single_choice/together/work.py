from . import clear
from . import sort
from . import generatejsonl

def main(paper_path, file_name):

    clear.main(paper_path, file_name)
    address_single_choice, address_multichoice, address_subject = sort.main(file_name)
    generatejsonl.main(address_single_choice, file_name)
    generatejsonl.main(address_multichoice, file_name)
    generatejsonl.main(address_subject, file_name)
