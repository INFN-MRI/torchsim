| backend | mode | threads | atoms | best (s) | atoms/s | peak RSS (MiB) | over baseline (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| epgpy | forward | 1 | 1 | 0.0240 | 42 | 36 | 4.8 |
| epgpy | forward | 1 | 10 | 0.0445 | 225 | 37 | 5.4 |
| epgpy | forward | 1 | 100 | 0.1508 | 663 | 43 | 11.7 |
| epgpy | forward | 1 | 1000 | 1.5767 | 634 | 109 | 77.1 |
| epgpy | forward | 1 | 10000 | 19.5585 | 511 | 761 | 729.8 |
| epgpy | jacobian | 1 | 1 | 0.0696 | 14 | 37 | 5.7 |
| epgpy | jacobian | 1 | 10 | 0.1461 | 68 | 39 | 7.3 |
| epgpy | jacobian | 1 | 100 | 0.6333 | 158 | 54 | 22.1 |
| epgpy | jacobian | 1 | 1000 | 7.1266 | 140 | 205 | 173.4 |
| epgpy | jacobian | 1 | 10000 | 103.2070 | 97 | 1716 | 1684.5 |
| sycomore | forward | 1 | 1 | 0.0014 | 719 | 33 | 0.5 |
| sycomore | forward | 1 | 10 | 0.0106 | 942 | 33 | 0.8 |
| sycomore | forward | 1 | 100 | 0.1464 | 683 | 35 | 2.9 |
| sycomore | forward | 1 | 100 | 0.1908 | 524 | 35 | 2.9 |
| sycomore | forward | 1 | 100 | 0.1327 | 754 | 35 | 2.8 |
| sycomore | forward | 1 | 100 | 0.1442 | 693 | 35 | 2.8 |
| sycomore | forward | 1 | 100 | 0.1448 | 691 | 35 | 2.8 |
| sycomore | forward | 1 | 100 | 0.1418 | 705 | 35 | 2.8 |
| sycomore | forward | 1 | 1000 | 1.4341 | 697 | 56 | 24.2 |
| sycomore | forward | 1 | 10000 | 14.3996 | 694 | 266 | 233.5 |
| torchsim | forward | 1 | 1000 | 0.5255 | 1,903 | 743 | 123.9 |
| torchsim | forward | 1 | 10000 | 5.2154 | 1,917 | 840 | 221.4 |
| torchsim | forward | 4 | 1 | 0.0015 | 656 | 726 | 107.1 |
| torchsim | forward | 4 | 10 | 0.0028 | 3,634 | 726 | 107.0 |
| torchsim | forward | 4 | 100 | 0.0178 | 5,614 | 726 | 107.0 |
| torchsim | forward | 4 | 1000 | 0.1436 | 6,962 | 738 | 119.7 |
| torchsim | forward | 4 | 10000 | 1.3538 | 7,387 | 840 | 221.4 |
| torchsim | forward | 4 | 100000 | 13.2164 | 7,566 | 1874 | 1255.0 |
| torchsim | jacobian | 4 | 1 | 0.0112 | 89 | 733 | 113.8 |
| torchsim | jacobian | 4 | 10 | 0.0259 | 386 | 733 | 113.8 |
| torchsim | jacobian | 4 | 100 | 0.1665 | 601 | 733 | 113.8 |
| torchsim | jacobian | 4 | 1000 | 1.4482 | 690 | 777 | 157.6 |
| torchsim | jacobian | 4 | 10000 | 14.3792 | 695 | 1076 | 457.6 |
| torchsim | jacobian | 4 | 100000 | 141.9735 | 704 | 4171 | 3551.9 |
| torchsim | jacobian(T1) | 4 | 100 | 0.0818 | 1,223 | 733 | 113.9 |
| torchsim | jacobian(T1) | 4 | 1000 | 0.7257 | 1,378 | 759 | 140.3 |
| torchsim | jacobian(T1) | 4 | 10000 | 7.1417 | 1,400 | 960 | 341.0 |

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
- sycomore n=100: threshold=0; orders reached 501-501
- sycomore n=100: threshold=0.01; orders reached 2-8
- sycomore n=100: threshold=0.001; orders reached 4-34
- sycomore n=100: threshold=0.0001; orders reached 6-75
- sycomore n=100: threshold=1e-06; orders reached 11-158
- sycomore n=1000: threshold=1e-06; orders reached 11-158
- sycomore n=10000: threshold=1e-06; orders reached 11-158
