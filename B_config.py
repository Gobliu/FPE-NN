# for B

# RUN_ID = 0
# T_GAP = 0.0001
# X_POINTS = 100
# X_MIN = 0.2
# X_MAX = 1.2

# RUN_ID = 1
# T_GAP = 0.0001
# RUN_ID = 2
# T_GAP = 0.00005
# RUN_ID = 3
# T_GAP = 0.00001
# RUN_ID = 5
# T_GAP = 0.0001

# RUN_ID = 4
# # T_GAP = 0.001
# X_POINTS = 100
# X_MIN = 0.01
# X_MAX = 0.11
#
# X_POINTS = 100
# X_MIN = 0.1
# X_MAX = 1.1

# RUN_ID = 6
# T_GAP = 0.0001
# X_POINTS = 100
# X_MIN = 0.5
# X_MAX = 1.5

# RUN_ID = 7
# T_GAP = 0.0002
# X_POINTS = 100
# X_MIN = 0.5
# X_MAX = 1.5

# RUN_ID = 8
# T_GAP = 0.0002
# X_POINTS = 100
# X_MIN = 0.1
# X_MAX = 1.1

#   ~~~~~~~~~~~~~~~~~~~~~~~ below multiple noise
# RUN_ID = 10
# T_GAP = 0.0001
# X_POINTS = 100
# X_MIN = 0.1
# X_MAX = 1.1
# RUN_ID = 12
# X_MIN = 0.1
# X_MAX = 1.5
# RUN_ID = 16             # ensure sum = 1, error 0.01
# RUN_ID = 17
# RUN_ID = 18               # no need sum = 100
# RUN_ID = 20
# T_GAP = 0.001
# X_POINTS = 100
# X_MIN = 0.01
# X_MAX = 1.01
# RUN_ID = 20
# T_GAP = 0.001
# X_POINTS = 100
# X_MIN = 0.1
# X_MAX = 1.1

# RUN_ID = 2012
# T_GAP = 0.001
# X_POINTS = 100
# X_MIN = 0.1
# X_MAX = 1.1
# T_POINTS = 50
# N_SAMPLE = 100

RUN_ID = 2016
T_GAP = 0.0005
X_POINTS = 100
X_MIN = 0.
X_MAX = 1
T_POINTS = 50
N_SAMPLE = 100

# RUN_ID = 2015
# T_GAP = 0.0002
# X_POINTS = 100
# X_MIN = 0.1
# X_MAX = 1.1
# T_POINTS = 50
# N_SAMPLE = 100

SEED = 19822012
# T_POINTS = 50
# T_POINTS = 25   # ID 7
# N_SAMPLE = 200  # ID 7
LEARNING_RATE_GH = 0.1**4
LEARNING_RATE_P = 2 * 0.1**3
EPOCH = 200000
BATCH_SIZE = 32
PATIENCE = 20
# SIGMA = 0.017     # ID 15
# SIGMA = 0.05
# SIGMA = 0.01      # ID 16
SIGMA = 0.015        # ID 17
# SIGMA = 0.07        # ID 20

# RUN_ID = 2
# X_POINTS = 100
# T_GAP = 0.00001
# X_MIN = 0.01
# X_MAX = 0.11
#
# SEED = 2015
# T_POINTS = 50
# N_SAMPLE = 100
# LEARNING_RATE_GH = 3*0.1**6    # for small OU, can try 3 or 4
# LEARNING_RATE_P = 0.1**3
# EPOCH = 200000
# BATCH_SIZE = 64
# PATIENCE = 20
# SIGMA = 0.5
