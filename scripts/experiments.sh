## MiniWorld

./kubernetes/run.sh d2_miniworld5_grid miniworld "baseline"

## MiniGrid

./kubernetes/run.sh dreamer2_maze11_top top_i1
./kubernetes/run.sh dreamer2_maze11_top top_i3
./kubernetes/run.sh dreamer2_maze11_top top_d2048_m2048_i1
./kubernetes/run.sh dreamer2_maze11_top top_d2048_m2048_i3

./kubernetes/run.sh dreamer2_maze11_map d2048_m2048_s100
./kubernetes/run.sh dreamer2_maze11_map d2048_m2048

./kubernetes/run.sh dreamer2_maze11_map d2048_m2048_i1
./kubernetes/run.sh dreamer2_maze11_map d2048_m2048_birnn512
./kubernetes/run.sh dreamer2_maze11_map d2048_m2048_i10
./kubernetes/run.sh dreamer2_maze11_map d2048_m2048_i3

./kubernetes/run.sh dreamer2_maze11_map d1024_i3_ndloss

./kubernetes/run.sh dreamer2_maze11_map d2048_m2048_i1
./kubernetes/run.sh dreamer2_maze11_map d1024_m2048_i1
./kubernetes/run.sh dreamer2_maze11_map d2048_m1024_i1
./kubernetes/run.sh dreamer2_maze11_map d1024_m1024_i1

./kubernetes/run.sh dreamer2_maze11_map d1024_m1024_i3
./kubernetes/run.sh dreamer2_maze11_map d512_m1024_i1
./kubernetes/run.sh dreamer2_maze11_map d1024_m1024_i1

./kubernetes/run.sh dreamer2_maze11_map d1024_i10
./kubernetes/run.sh dreamer2_maze11_map d1024_i3
./kubernetes/run.sh dreamer2_maze11_map d1024_i1

./kubernetes/run.sh dreamer2_maze11_map deter1024_1024
./kubernetes/run.sh dreamer2_maze11_map deter2048

./kubernetes/run.sh dreamer2_maze11_map maprnn_2048
./kubernetes/run.sh dreamer2_maze11_map maprnn_1024
./kubernetes/run.sh dreamer2_maze11_map deter1024
./kubernetes/run.sh dreamer2_maze11_map maprnn

./kubernetes/run.sh dreamer2_maze11_map iwae10_deter1024
./kubernetes/run.sh dreamer2_maze11_map iwae3_deter2048
./kubernetes/run.sh dreamer2_maze11_map iwae3_deter1024
./kubernetes/run.sh dreamer2_maze11_map iwae10
./kubernetes/run.sh dreamer2_maze11_map iwae3_birnn
./kubernetes/run.sh dreamer2_maze11_map iwae3
./kubernetes/run.sh dreamer2_maze11_map baseline_100m

##

./kubernetes/run.sh dreamer2_maze11_top maprnn_top

./kubernetes/run.sh dreamer2_maze11_top birnn_i3_imagine
./kubernetes/run.sh dreamer2_maze11_top iwae3_imagine

./kubernetes/run.sh dreamer2_maze11_top birnn_i3_fixfulleval

./kubernetes/run.sh dreamer2_maze11_top birnn_i3_fixlogprob
./kubernetes/run.sh dreamer2_maze11_top iwae3_fixlogprob
./kubernetes/run.sh dreamer2_maze11_top iwae3_fixloss
./kubernetes/run.sh dreamer2_maze11_top birnn_i3_fixloss

./kubernetes/run.sh dreamer2_maze11_top debug_lossmodel

./kubernetes/run.sh dreamer2_maze11_top iwae3_eval_refactor

./kubernetes/run.sh dreamer2_maze11_top iwae3_batch64
./kubernetes/run.sh dreamer2_maze11_top iwae3_minprob

./kubernetes/run.sh dreamer2_maze11_top iwae1_refactor
./kubernetes/run.sh dreamer2_maze11_top iwae3
./kubernetes/run.sh dreamer2_maze11_top iwae10

./kubernetes/run.sh dreamer2_maze11_top minprob_logmore
./kubernetes/run.sh dreamer2_maze11_top minprob1e-2_logbatch
./kubernetes/run.sh dreamer2_maze11_top debug_logbatch

./kubernetes/run.sh dreamer2_maze11_top iwae1

./kubernetes/run.sh dreamer2_maze11_top minprob1e-3
./kubernetes/run.sh dreamer2_maze11_top minprob1e-5
./kubernetes/run.sh dreamer2_maze11_top minprob1e-9

./kubernetes/run.sh dreamer2_maze11_top batch128
./kubernetes/run.sh dreamer2_maze11_top batch64
./kubernetes/run.sh dreamer2_maze11_top default_deter512_dec2

./kubernetes/run.sh dreamer2_maze11_top bigger_deter_dec2layer_mapgrad
./kubernetes/run.sh dreamer2_maze11_top bigger_deter_dec2layer
./kubernetes/run.sh dreamer2_maze11_top input_rnn

./kubernetes/run.sh dreamer2_maze11_top map_grad      # Flow gradients to map - good map, but does it improve model?
./kubernetes/run.sh dreamer2_maze11_top bigger_stoch  # Bigger stochastic - improves map prediction?
./kubernetes/run.sh dreamer2_maze11_top bigger_deter  # Bigger deterministic - improves map prediction?
./kubernetes/run.sh dreamer2_maze11_top dec2layer  # Maybe easier for map decoder?

./kubernetes/run.sh dreamer2_maze11_top top_100m #
./kubernetes/run.sh dreamer2_maze11_top map_rnn_100m

./kubernetes/run.sh dreamer2_maze11_map dense_encoder3
./kubernetes/run.sh dreamer2_maze11_map dense_encoder

./kubernetes/run.sh dreamer2_maze11_map kl01
./kubernetes/run.sh dreamer2_maze11_map kl03
./kubernetes/run.sh dreamer2_maze11_map kl05

./kubernetes/run.sh dreamer2_maze11_map b100

./kubernetes/run.sh dreamer2_maze11_map perf_cudnnbenchmark

./kubernetes/run.sh dreamer2_maze11_map lr10
./kubernetes/run.sh dreamer2_maze11_map lr5
./kubernetes/run.sh dreamer2_maze11_map lr3

./kubernetes/run.sh dreamer2_debug adam && ./kubernetes/run.sh dreamer2_debug adam

./kubernetes/run.sh dreamer2_debug adam_eps7_lr5
./kubernetes/run.sh dreamer2_debug adam_eps7

./kubernetes/run.sh dreamer2_debug sgd_01

./kubernetes/run.sh dreamer2_debug rmsprop_eps7
./kubernetes/run.sh dreamer2_debug adam
./kubernetes/run.sh dreamer2_debug rmsprop

./kubernetes/run.sh dreamer2_debug adam_noeps
./kubernetes/run.sh dreamer2_debug rmsprop
./kubernetes/run.sh dreamer2_debug tfdebug

./kubernetes/run.sh dreamer2_maze11_map tfmatch_eps10
./kubernetes/run.sh dreamer2_maze11_map tfmatch_eps30

./kubernetes/run.sh dreamer2_maze11_map baseline_15m
