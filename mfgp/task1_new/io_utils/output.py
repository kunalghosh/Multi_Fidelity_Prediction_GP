import time
import datetime
import multiprocessing as multi
import contextlib


@contextlib.contextmanager
def log_timing(conf, description):
    start = time.time()
    append_write(conf.out_name, f"START: {description}\n")
    ### actual piece of code to time
    yield
    ### 
    append_write(conf.out_name, f"FINISH: {description}\n")
    process_time = time.time() - start
    out_time(conf.out_name, process_time)

def out_condition(filepath, InData):
    """                                                                                                                                                                       

    """
    f = open(filepath, 'a')
    f.write("=============================" + "\n")
    dt_now = datetime.datetime.now()
    f.write(str(dt_now) + "\n" )
    f.write("condition \n" )
    f.write("fn_name(acquistion function) " + InData.fn_name + "\n" )
    f.write("num_itr " + str(InData.num_itr) + "\n" )
    f.write("dataset_size " + str(InData.dataset_size) + "\n" )
    f.write("test_set_size " + str(InData.test_set_size) + "\n" )
    f.write("rnd_size(random sampling) or K_pre(high_and_cluster) " + str(InData.rnd_size) + "\n" )#
    f.write("high_size(high std) " + str(InData.K_high) + "\n" )
    f.write("flag for reducing the MBTR array " + str(InData.mbtr_red) + "\n" )
    f.write("mbtr_path " + InData.mbtr_path + "\n" )
    f.write("json_path " + InData.json_path + "\n" )
    f.write("save_load_flag " + InData.save_load_flag + "\n" )
    f.write("loadidxs_path " + InData.loadidxs_path + "\n" )
    f.write("save_load_split_flag " + str(InData.save_load_split_flag) + "\n" )
    f.write("loadsplitidxs_path " + InData.loadsplitidxs_path + "\n" )
    f.write("CPU: " + str(multi.cpu_count()) + "\n" )
    f.write("dataset is " + str(InData.dataset) + "\n" )
    for i in range(InData.num_itr + 1):
        f.write( "the number of added indexes in step " +str(i) + " " + str(InData.pre_idxs[i]) + "\n" )   
    f.write("preprocess " + InData.preprocess + "\n" )
    f.write("length_value " + str(InData.length) + "\n" )
    f.write("const_value " + str(InData.const) + "\n" )
    f.write("upper bound " + str(InData.bound) + "\n" )
    f.write("lower bound " + str(1.0/InData.bound) + "\n" )
    f.write("n_restart_optimizer " + str(InData.n_opt) + "\n" )
    f.flush()
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

def out_time(filepath, time):
    """                                                                                                
    Writes calculation time into the given file appending.                                                                                                           
    """
    f = open(filepath, 'a')
    f.write("time "),f.write(str(time) + "[s]" + "\n")
    f.close()

def out_time_all(filepath, time):
    """                                                                                                
    Writes calculation time into the given file appending.                                                                                                           
    """
    f = open(filepath, 'a')
    f.write("time all "),f.write(str(time) + "[s]" + "\n")
    f.close()
    
def out_time_all_temp(filepath, time):
    """                                                                                                
    Writes calculation time into the given file appending.                                                                                                           
    """
    f = open(filepath, 'a')
    f.write("time for this step "),f.write(str(time) + "[s]" + "\n")
    f.close()    
