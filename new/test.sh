#!/usr/bin/env bash

#!/usr/bin/env bash

python3 make_dataset.py -p 5 -s 1 -d 1 -t erdos --folder test
python3 run_experiments.py -n 32 -b 1 -k 2 --boot 50 -s 2 --folder test --strategy entropy-enum
