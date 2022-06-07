import logging

# date_strftime_format = "%d-%b-%y %H:%M:%S"
logging.basicConfig(filename = '/'.join(__file__.split('/')[:-1]) + '/logs/log.txt', 
                    filemode = 'a',
                    datefmt = "%H:%M:%S", 
                    level = logging.INFO,
                    format = '%(asctime)s[%(levelname)s] ~ %(message)s')

def write_log(info = None, c_tag = None, verbose = True):
    if verbose:
        print(str(info))
        
    if c_tag is not None:
        logging.info(str(c_tag) + ' ' + str(info))
    else:
         logging.info(str(info))
    

write_log('\n\n' + "**" * 40 + '\n' + " " * 37 + 'new log\n' + "**" * 40)
