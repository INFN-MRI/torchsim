| backend | mode | threads | atoms | best (s) | atoms/s | peak RSS (MiB) | over baseline (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BlochSimulators.jl | forward | 1 | 1000 | 0.0847 | 11,808 | 808 | 145.6 |
| BlochSimulators.jl | forward | 1 | 10000 | 0.8555 | 11,690 | 843 | 181.7 |
| BlochSimulators.jl | forward | 1 | 100000 | 9.0561 | 11,042 | 1181 | 521.0 |
| BlochSimulators.jl | forward | 4 | 1 | 0.0003 | 3,056 | 819 | 173.2 |
| BlochSimulators.jl | forward | 4 | 10 | 0.0010 | 9,570 | 818 | 173.2 |
| BlochSimulators.jl | forward | 4 | 100 | 0.0046 | 21,822 | 832 | 188.1 |
| BlochSimulators.jl | forward | 4 | 1000 | 0.0242 | 41,305 | 824 | 183.3 |
| BlochSimulators.jl | forward | 4 | 10000 | 0.2526 | 39,581 | 870 | 225.4 |
| BlochSimulators.jl | forward | 4 | 100000 | 2.5157 | 39,750 | 1187 | 545.8 |
| BlochSimulators.jl | forward(complex) | 4 | 1000 | 0.1005 | 9,951 | 799 | 153.7 |
| BlochSimulators.jl | forward(complex) | 4 | 10000 | 0.9701 | 10,308 | 825 | 178.6 |
| BlochSimulators.jl | forward(complex) | 4 | 100000 | 9.5250 | 10,499 | 1617 | 970.5 |
| BlochSimulators.jl | jacobian(T1,T2) | 4 | 1000 | 0.0813 | 12,307 | 844 | 198.8 |
| BlochSimulators.jl | jacobian(T1,T2) | 4 | 10000 | 0.7736 | 12,927 | 954 | 308.7 |
| BlochSimulators.jl | jacobian(T1,T2) | 4 | 100000 | 8.5213 | 11,735 | 1900 | 1258.0 |
| KomaMRI.jl | forward | 4 | 1 | 0.1174 | 9 | 898 | 30.9 |
| KomaMRI.jl | forward | 4 | 10 | 0.4169 | 24 | 945 | 82.4 |
| KomaMRI.jl | forward | 4 | 100 | 1.0960 | 91 | 1373 | 504.7 |
| epgpy | forward | 1 | 1 | 0.0339 | 30 | 36 | 4.5 |
| epgpy | forward | 1 | 10 | 0.0665 | 150 | 36 | 5.0 |
| epgpy | forward | 1 | 100 | 0.2503 | 400 | 43 | 11.9 |
| epgpy | forward | 1 | 1000 | 2.3033 | 434 | 110 | 79.1 |
| epgpy | forward | 1 | 10000 | 36.3704 | 275 | 783 | 751.4 |
| epgpy | jacobian(T1,T2) | 1 | 1 | 0.0904 | 11 | 37 | 5.4 |
| epgpy | jacobian(T1,T2) | 1 | 10 | 0.2378 | 42 | 38 | 6.8 |
| epgpy | jacobian(T1,T2) | 1 | 100 | 1.3470 | 74 | 54 | 22.5 |
| epgpy | jacobian(T1,T2) | 1 | 1000 | 13.9996 | 71 | 209 | 177.6 |
| epgpy | jacobian(T1,T2) | 1 | 10000 | 177.0630 | 56 | 1759 | 1728.0 |
| sycomore | forward | 1 | 1 | 0.0020 | 494 | 32 | 0.6 |
| sycomore | forward | 1 | 10 | 0.0243 | 412 | 33 | 0.8 |
| sycomore | forward | 1 | 100 | 0.2437 | 410 | 35 | 3.0 |
| sycomore | forward | 1 | 1000 | 2.5105 | 398 | 56 | 23.8 |
| sycomore | forward | 1 | 10000 | 24.0942 | 415 | 265 | 233.5 |
| torchsim | forward | 1 | 1000 | 0.0670 | 14,925 | 723 | 116.5 |
| torchsim | forward | 1 | 10000 | 0.6841 | 14,618 | 827 | 219.9 |
| torchsim | forward | 4 | 1 | 0.0019 | 522 | 714 | 107.0 |
| torchsim | forward | 4 | 10 | 0.0043 | 2,339 | 714 | 107.0 |
| torchsim | forward | 4 | 100 | 0.0282 | 3,551 | 714 | 107.1 |
| torchsim | forward | 4 | 1000 | 0.0221 | 45,284 | 723 | 116.4 |
| torchsim | forward | 4 | 10000 | 0.1839 | 54,385 | 826 | 218.7 |
| torchsim | forward | 4 | 100000 | 23.4589 | 4,263 | 1862 | 1255.1 |
| torchsim | jacobian(T1) | 4 | 1 | 0.0089 | 112 | 721 | 113.8 |
| torchsim | jacobian(T1) | 4 | 10 | 0.0229 | 437 | 721 | 113.7 |
| torchsim | jacobian(T1) | 4 | 100 | 0.1488 | 672 | 721 | 113.7 |
| torchsim | jacobian(T1) | 4 | 1000 | 0.4515 | 2,215 | 746 | 138.4 |
| torchsim | jacobian(T1) | 4 | 10000 | 1.5669 | 6,382 | 948 | 340.7 |
| torchsim | jacobian(T1) | 4 | 100000 | 152.7505 | 655 | 3012 | 2404.9 |
| torchsim | jacobian(T1,T2) | 4 | 1 | 0.0166 | 60 | 721 | 113.7 |
| torchsim | jacobian(T1,T2) | 4 | 10 | 0.0443 | 226 | 721 | 113.7 |
| torchsim | jacobian(T1,T2) | 4 | 100 | 0.2971 | 337 | 721 | 113.7 |
| torchsim | jacobian(T1,T2) | 4 | 1000 | 0.9038 | 1,106 | 760 | 152.7 |
| torchsim | jacobian(T1,T2) | 4 | 10000 | 3.1090 | 3,217 | 1061 | 454.2 |
| torchsim | jacobian(T1,T2) | 4 | 100000 | 278.2875 | 359 | 4158 | 3550.8 |

- BlochSimulators.jl n=1000: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=10000: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=100000: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=1: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=10: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=100: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=1000: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=10000: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=100000: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=1000: float32; complex RF train, complex states; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=10000: float32; complex RF train, complex states; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=100000: float32; complex RF train, complex states; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=1000: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction; forward differences, three passes
- BlochSimulators.jl n=10000: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction; forward differences, three passes
- BlochSimulators.jl n=100000: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction; forward differences, three passes
- KomaMRI.jl n=1: isochromat: 64 spins per tissue through one spoiler cycle, 64 spins in all
- KomaMRI.jl n=10: isochromat: 64 spins per tissue through one spoiler cycle, 640 spins in all
- KomaMRI.jl n=100: isochromat: 64 spins per tissue through one spoiler cycle, 6400 spins in all
- epgpy n=1: max_nstate matches the orders TorchSim keeps
- epgpy n=10: max_nstate matches the orders TorchSim keeps
- epgpy n=100: max_nstate matches the orders TorchSim keeps
- epgpy n=1000: max_nstate matches the orders TorchSim keeps
- epgpy n=10000: max_nstate matches the orders TorchSim keeps
- epgpy n=1: max_nstate matches the orders TorchSim keeps
- epgpy n=10: max_nstate matches the orders TorchSim keeps
- epgpy n=100: max_nstate matches the orders TorchSim keeps
- epgpy n=1000: max_nstate matches the orders TorchSim keeps
- epgpy n=10000: max_nstate matches the orders TorchSim keeps
- sycomore n=1: threshold=1e-06; orders reached 11-11
- sycomore n=10: threshold=1e-06; orders reached 11-158
- sycomore n=100: threshold=1e-06; orders reached 11-158
- sycomore n=1000: threshold=1e-06; orders reached 11-158
- sycomore n=10000: threshold=1e-06; orders reached 11-158
