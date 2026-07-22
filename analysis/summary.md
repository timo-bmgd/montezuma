# Run analysis summary

| run | algo | tv_mode | max_rooms | left_room1_step | final_return_l20 | max_return | peak_intrinsic | peak_intrinsic_step | final_entropy_l20 | final_tv_share_l20 | final_expl_var_l20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_tv_off s1 | ppo | off | 2 | 3.99M | 0 | 400.0 |  | never | 2.763 |  | -3.332 |
| rnd_tv_off s1 | rnd | off | 1 | never | 0 | 0.0 | 192.832 | 0.00M | 0.503 | -0.091 | -6.764 |
| rnd_tv_off s2 | rnd | off | 2 | 6.44M | 10.0 | 400.0 | 145.341 | 0.00M | 0.611 | -0.198 | 0.626 |
| rnd_tv_remote s1 | rnd | remote | 1 | never | 0 | 100.0 | 257.918 | 0.00M | 0.762 | 0.117 | -0.262 |
| rnd_tv_sham-remote s1 | rnd | sham-remote | 1 | never | 0 | 0.0 | 218.954 | 0.00M | 0.415 | -0.063 | -6.358 |
| rnd_tv_static s1 | rnd | static | 1 | never | 0 | 0.0 | 234.357 | 0.00M | 0.282 | 0.203 | -0.898 |
