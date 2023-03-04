import sys
from fwi_func import get_codes_ds

fpath = sys.argv[1]
write_to = sys.argv[2]
get_codes_ds(fpath, write_to=write_to)
