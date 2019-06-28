import sys
import csv
import pandas as pd

# get the path to the pandas df (JSON file) containing the data.
json_path = sys.argv[1]
out_filename = sys.argv[2]

df_62k = pd.read_json(json_path, orient='split')
# with open(out_filename, 'w') as f:
#     df_62k.to_string(f)

# exit(0)
print("writing csv now.")
df_62k[:10].to_csv(
    out_filename,
    columns=['xyz_pbe_relaxed'],
    header=False,  # do not print column names
    index=False,  # do not print row numbers
    # quotechar=' ',  # do not quote
    quoting=csv.QUOTE_MINIMAL,  # do not quote
    # quoting=csv.QUOTE_NONE,  # do not quote
    # sep=' '  # don't use a separator
)
