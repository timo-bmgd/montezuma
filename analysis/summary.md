# Run analysis summary

| run | tv_mode | n_episodes | room2_episodes | max_rooms | left_room1_step | final_return_l20 | max_return | peak_intrinsic | final_entropy_l20 | final_tv_share_l20 | final_expl_var_l20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ppo_tv_off s1 | off | 18888 | 2 | 2 | 3.99M | 0 | 400.0 |  | 2.763 |  | -3.332 |
| rnd_tv_off s1 | off | 19267 | 0 | 1 | never | 0 | 0.0 | 192.832 | 0.503 | -0.091 | -6.764 |
| rnd_tv_off s2 | off | 18244 | 6 | 2 | 6.44M | 10.0 | 400.0 | 145.341 | 0.611 | -0.198 | 0.626 |
| rnd_tv_remote s1 | remote | 20640 | 0 | 1 | never | 0 | 100.0 | 257.918 | 0.762 | 0.117 | -0.262 |
| rnd_tv_sham-remote s1 | sham-remote | 18802 | 0 | 1 | never | 0 | 0.0 | 218.954 | 0.415 | -0.063 | -6.358 |
| rnd_tv_static s1 | static | 22225 | 0 | 1 | never | 0 | 0.0 | 234.357 | 0.282 | 0.203 | -0.898 |
