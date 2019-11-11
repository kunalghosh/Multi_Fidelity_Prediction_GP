def out_condition(filepath, list_parameter):
    """                                                                                                                                                                       
    Writes the text into the given file overwriting.                                                                                                                          
    """
    f = open(filepath, 'w')
    f.write(text)
    f.close()

def overwrite(filepath, text):
    """                                                                                                                                                                       
    Writes the text into the given file overwriting.                                                                                                                          
    """
    f = open(filepath, 'w')
    f.write(text)
    f.close()

def append_write(filepath, text):
    """                                                                                                                
    Writes the text into the given file appending.                                                                                                                           
    """
    f = open(filepath, 'a')
    f.write(text)
    f.close()

