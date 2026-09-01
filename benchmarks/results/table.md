| backend | mode | device | threads | atoms | best (s) | atoms/s | peak RSS (MiB) | over baseline (MiB) | device (MiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BlochSimulators.jl | forward | cpu | 1 | 1000 | 0.0802 | 12,466 | 1304 | 171.5 | 0 |
| BlochSimulators.jl | forward | cpu | 1 | 10000 | 0.8611 | 11,614 | 1334 | 198.6 | 0 |
| BlochSimulators.jl | forward | cpu | 1 | 100000 | 9.1049 | 10,983 | 1808 | 674.6 | 0 |
| BlochSimulators.jl | forward | cpu | 4 | 1 | 0.0002 | 4,294 | 1322 | 187.6 | 0 |
| BlochSimulators.jl | forward | cpu | 4 | 10 | 0.0010 | 10,154 | 1306 | 172.0 | 0 |
| BlochSimulators.jl | forward | cpu | 4 | 100 | 0.0021 | 47,610 | 1301 | 165.4 | 0 |
| BlochSimulators.jl | forward | cpu | 4 | 1000 | 0.0217 | 46,007 | 1300 | 169.2 | 0 |
| BlochSimulators.jl | forward | cpu | 4 | 10000 | 0.2116 | 47,251 | 1328 | 194.7 | 0 |
| BlochSimulators.jl | forward | cpu | 4 | 100000 | 2.2758 | 43,941 | 1886 | 752.6 | 0 |
| BlochSimulators.jl | forward | cuda | 4 | 1 | 0.0002 | 6,378 | 1394 | 248.4 | 32 |
| BlochSimulators.jl | forward | cuda | 4 | 10 | 0.0002 | 63,031 | 1370 | 234.8 | 128 |
| BlochSimulators.jl | forward | cuda | 4 | 100 | 0.0002 | 649,916 | 1383 | 245.7 | 1568 |
| BlochSimulators.jl | forward | cuda | 4 | 1000 | 0.0020 | 496,652 | 1372 | 230.8 | 3552 |
| BlochSimulators.jl | forward | cuda | 4 | 10000 | 0.0044 | 2,278,960 | 1391 | 255.3 | 3552 |
| BlochSimulators.jl | forward | cuda | 4 | 100000 | 0.0372 | 2,690,142 | 1577 | 442.6 | 3648 |
| BlochSimulators.jl | forward(complex) | cpu | 4 | 1000 | 0.0424 | 23,594 | 1280 | 137.2 | 0 |
| BlochSimulators.jl | forward(complex) | cpu | 4 | 10000 | 0.5203 | 19,221 | 1384 | 250.6 | 0 |
| BlochSimulators.jl | forward(complex) | cpu | 4 | 100000 | 5.4241 | 18,436 | 2471 | 1339.8 | 0 |
| BlochSimulators.jl | forward(complex) | cuda | 4 | 1000 | 0.0038 | 264,849 | 1375 | 239.2 | 3552 |
| BlochSimulators.jl | forward(complex) | cuda | 4 | 10000 | 0.0110 | 908,740 | 1405 | 272.2 | 3552 |
| BlochSimulators.jl | forward(complex) | cuda | 4 | 100000 | 0.0886 | 1,128,364 | 1728 | 589.2 | 4960 |
| BlochSimulators.jl | jacobian(T1,T2) | cpu | 4 | 1000 | 0.0698 | 14,327 | 1307 | 178.5 | 0 |
| BlochSimulators.jl | jacobian(T1,T2) | cpu | 4 | 10000 | 0.8471 | 11,805 | 1461 | 328.6 | 0 |
| BlochSimulators.jl | jacobian(T1,T2) | cpu | 4 | 100000 | 7.4086 | 13,498 | 2633 | 1502.4 | 0 |
| BlochSimulators.jl | jacobian(T1,T2) | cuda | 4 | 1000 | 0.0014 | 694,170 | 1402 | 267.8 | 3552 |
| BlochSimulators.jl | jacobian(T1,T2) | cuda | 4 | 10000 | 0.0118 | 850,555 | 1391 | 253.2 | 3584 |
| BlochSimulators.jl | jacobian(T1,T2) | cuda | 4 | 100000 | 0.1189 | 841,265 | 1573 | 439.4 | 3872 |
| KomaMRI.jl | forward | cpu | 4 | 1 | 0.0892 | 11 | 930 | 76.2 | 0 |
| KomaMRI.jl | forward | cpu | 4 | 10 | 0.1203 | 83 | 1017 | 150.8 | 0 |
| KomaMRI.jl | forward | cpu | 4 | 100 | 0.8882 | 113 | 1502 | 637.4 | 0 |
| KomaMRI.jl | forward | cpu | 4 | 1000 | 7.3347 | 136 | 5850 | 4998.1 | 0 |
| KomaMRI.jl | forward | cuda | 4 | 1 | 0.9776 | 1 | 2004 | 484.3 | 96 |
| KomaMRI.jl | forward | cuda | 4 | 10 | 1.0743 | 9 | 1921 | 396.0 | 352 |
| KomaMRI.jl | forward | cuda | 4 | 100 | 1.1945 | 84 | 2085 | 572.8 | 960 |
| KomaMRI.jl | forward | cuda | 4 | 1000 | 2.6255 | 381 | 3096 | 1575.4 | 5248 |
| epgpy | forward | cpu | 1 | 1 | 0.0292 | 34 | 37 | 4.5 | 0 |
| epgpy | forward | cpu | 1 | 10 | 0.0533 | 188 | 37 | 4.8 | 0 |
| epgpy | forward | cpu | 1 | 100 | 0.2371 | 422 | 44 | 11.9 | 0 |
| epgpy | forward | cpu | 1 | 1000 | 2.5102 | 398 | 111 | 79.1 | 0 |
| epgpy | forward | cpu | 1 | 10000 | 34.7414 | 288 | 783 | 751.2 | 0 |
| epgpy | jacobian(T1,T2) | cpu | 1 | 1 | 0.0862 | 12 | 38 | 5.3 | 0 |
| epgpy | jacobian(T1,T2) | cpu | 1 | 10 | 0.1852 | 54 | 39 | 6.9 | 0 |
| epgpy | jacobian(T1,T2) | cpu | 1 | 100 | 0.9521 | 105 | 56 | 23.4 | 0 |
| epgpy | jacobian(T1,T2) | cpu | 1 | 1000 | 10.5519 | 95 | 210 | 177.5 | 0 |
| epgpy | jacobian(T1,T2) | cpu | 1 | 10000 | 155.9514 | 64 | 1760 | 1727.5 | 0 |
| sycomore | forward | cpu | 1 | 1 | 0.0023 | 439 | 33 | 0.3 | 0 |
| sycomore | forward | cpu | 1 | 10 | 0.0268 | 373 | 34 | 0.8 | 0 |
| sycomore | forward | cpu | 1 | 100 | 0.2687 | 372 | 37 | 3.5 | 0 |
| sycomore | forward | cpu | 1 | 1000 | 2.2262 | 449 | 57 | 23.8 | 0 |
| sycomore | forward | cpu | 1 | 10000 | 28.3982 | 352 | 266 | 233.0 | 0 |
| torchsim | forward | cpu | 1 | 1000 | 0.0698 | 14,324 | 724 | 117.5 | 0 |
| torchsim | forward | cpu | 1 | 10000 | 0.7669 | 13,039 | 822 | 215.4 | 0 |
| torchsim | forward | cpu | 1 | 100000 | 7.1735 | 13,940 | 1861 | 1254.3 | 0 |
| torchsim | forward | cpu | 4 | 1 | 0.0009 | 1,132 | 708 | 101.6 | 0 |
| torchsim | forward | cpu | 4 | 10 | 0.0016 | 6,343 | 708 | 101.1 | 0 |
| torchsim | forward | cpu | 4 | 100 | 0.0037 | 27,010 | 708 | 101.2 | 0 |
| torchsim | forward | cpu | 4 | 1000 | 0.0233 | 42,972 | 720 | 113.5 | 0 |
| torchsim | forward | cpu | 4 | 10000 | 0.1866 | 53,578 | 860 | 252.6 | 0 |
| torchsim | forward | cpu | 4 | 100000 | 1.9028 | 52,554 | 1861 | 1254.0 | 0 |
| torchsim | forward | cuda | 4 | 1 | 0.0037 | 267 | 1152 | 551.3 | 54 |
| torchsim | forward | cuda | 4 | 10 | 0.0034 | 2,942 | 1180 | 582.2 | 54 |
| torchsim | forward | cuda | 4 | 100 | 0.0028 | 35,256 | 1240 | 639.7 | 54 |
| torchsim | forward | cuda | 4 | 1000 | 0.0033 | 301,734 | 1202 | 600.6 | 54 |
| torchsim | forward | cuda | 4 | 10000 | 0.0087 | 1,150,290 | 1194 | 592.4 | 196 |
| torchsim | forward | cuda | 4 | 100000 | 0.0687 | 1,454,714 | 1204 | 602.3 | 1586 |
| torchsim | jacobian(T1) | cpu | 4 | 1 | 0.0054 | 185 | 712 | 105.2 | 0 |
| torchsim | jacobian(T1) | cpu | 4 | 10 | 0.0081 | 1,234 | 712 | 105.4 | 0 |
| torchsim | jacobian(T1) | cpu | 4 | 100 | 0.0117 | 8,520 | 713 | 106.3 | 0 |
| torchsim | jacobian(T1) | cpu | 4 | 1000 | 0.0534 | 18,738 | 745 | 138.4 | 0 |
| torchsim | jacobian(T1) | cpu | 4 | 10000 | 0.5511 | 18,146 | 941 | 333.8 | 0 |
| torchsim | jacobian(T1) | cpu | 4 | 100000 | 5.4812 | 18,244 | 3007 | 2400.2 | 0 |
| torchsim | jacobian(T1) | cuda | 4 | 1 | 0.0084 | 119 | 1167 | 560.0 | 122 |
| torchsim | jacobian(T1) | cuda | 4 | 10 | 0.0081 | 1,242 | 1158 | 556.2 | 54 |
| torchsim | jacobian(T1) | cuda | 4 | 100 | 0.0079 | 12,640 | 1209 | 607.9 | 72 |
| torchsim | jacobian(T1) | cuda | 4 | 1000 | 0.0080 | 124,412 | 1211 | 609.8 | 56 |
| torchsim | jacobian(T1) | cuda | 4 | 10000 | 0.0237 | 421,684 | 1206 | 605.2 | 316 |
| torchsim | jacobian(T1) | cuda | 4 | 100000 | 0.1895 | 527,615 | 1209 | 607.7 | 2732 |
| torchsim | jacobian(T1,T2) | cpu | 4 | 1 | 0.0101 | 99 | 712 | 105.1 | 0 |
| torchsim | jacobian(T1,T2) | cpu | 4 | 10 | 0.0117 | 856 | 712 | 105.7 | 0 |
| torchsim | jacobian(T1,T2) | cpu | 4 | 100 | 0.0229 | 4,358 | 714 | 107.2 | 0 |
| torchsim | jacobian(T1,T2) | cpu | 4 | 1000 | 0.1285 | 7,783 | 757 | 150.1 | 0 |
| torchsim | jacobian(T1,T2) | cpu | 4 | 10000 | 1.0879 | 9,192 | 1055 | 448.1 | 0 |
| torchsim | jacobian(T1,T2) | cpu | 4 | 100000 | 9.6565 | 10,356 | 4147 | 3540.4 | 0 |
| torchsim | jacobian(T1,T2) | cuda | 4 | 1 | 0.0156 | 64 | 1164 | 563.0 | 72 |
| torchsim | jacobian(T1,T2) | cuda | 4 | 10 | 0.0156 | 640 | 1197 | 595.8 | 54 |
| torchsim | jacobian(T1,T2) | cuda | 4 | 100 | 0.0159 | 6,282 | 1253 | 651.3 | 54 |
| torchsim | jacobian(T1,T2) | cuda | 4 | 1000 | 0.0158 | 63,392 | 1206 | 605.0 | 56 |
| torchsim | jacobian(T1,T2) | cuda | 4 | 10000 | 0.0459 | 217,764 | 1206 | 605.3 | 512 |
| torchsim | jacobian(T1,T2) | cuda | 4 | 100000 | 0.3813 | 262,295 | 1208 | 606.0 | 4642 |

- BlochSimulators.jl n=1000: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=10000: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=100000: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=1: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=10: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=100: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction
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
- BlochSimulators.jl n=1000: float32; complex RF train, complex states; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=10000: float32; complex RF train, complex states; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=100000: float32; complex RF train, complex states; max_state is a multiple of 32 by construction
- BlochSimulators.jl n=1000: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction; forward differences, three passes
- BlochSimulators.jl n=10000: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction; forward differences, three passes
- BlochSimulators.jl n=100000: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction; forward differences, three passes
- BlochSimulators.jl n=1000: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction; forward differences, three passes
- BlochSimulators.jl n=10000: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction; forward differences, three passes
- BlochSimulators.jl n=100000: float32; real RF train, so the states stay real; max_state is a multiple of 32 by construction; forward differences, three passes
- KomaMRI.jl n=1: isochromat: 64 spins per tissue through one spoiler cycle, 64 spins in all
- KomaMRI.jl n=10: isochromat: 64 spins per tissue through one spoiler cycle, 640 spins in all
- KomaMRI.jl n=100: isochromat: 64 spins per tissue through one spoiler cycle, 6400 spins in all
- KomaMRI.jl n=1000: isochromat: 64 spins per tissue through one spoiler cycle, 64000 spins in all
- KomaMRI.jl n=1: isochromat: 64 spins per tissue through one spoiler cycle, 64 spins in all
- KomaMRI.jl n=10: isochromat: 64 spins per tissue through one spoiler cycle, 640 spins in all
- KomaMRI.jl n=100: isochromat: 64 spins per tissue through one spoiler cycle, 6400 spins in all
- KomaMRI.jl n=1000: isochromat: 64 spins per tissue through one spoiler cycle, 64000 spins in all
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
