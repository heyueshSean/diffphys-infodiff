# MANIFEST — 字节级 provenance

由 `python tools/make_manifest.py` 生成。本仓库的每一份代码都按 **sha1** 归档：
训练机上发车前必须核对 sha1 与行数，否则会「跑了旧版却以为改过了」。
（换行约定：B 线/C 线 = LF，A 线 = CRLF，这是各线的历史约定，不要改动。）

| 文件 | sha1 | 字节 | 行数 | 换行 |
|---|---|---:|---:|---|
| `rl/README.md` | `fb3edef51335764ee7ae9e0e73c5500ad9ccf02a` | 3029 | 51 | LF |
| `rl/breakout/physics_guided_ppo_breakout_v10.py` | `c55fe6c404a04d5b6210be429cf61fc6f66492ab` | 26933 | 591 | LF |
| `rl/breakout/physics_guided_ppo_breakout_v13.py` | `89b5703c1c603ec1cabca1512bb4b9baed326cce` | 34020 | 724 | LF |
| `rl/breakout/physics_guided_ppo_breakout_v16.py` | `1bde24bd14612aa65c0bfe276de2467ec6b933a3` | 38199 | 818 | LF |
| `rl/breakout/physics_guided_ppo_breakout_v21.py` | `fa0abffd69bf85c0c2f80619d1dd5c75eb066bd1` | 44136 | 952 | LF |
| `rl/breakout/physics_guided_ppo_breakout_v24.py` | `f2730b14af4665c9b2aa0a0890d30113596836bf` | 49122 | 1013 | LF |
| `rl/breakout/physics_guided_ppo_breakout_v26.py` | `a710237b9f1d0a7fa88093e1a7ab665d897ecedd` | 51218 | 1065 | LF |
| `rl/breakout/physics_guided_ppo_breakout_v30.py` | `157836c0d52a335524aee67470326b9609869daf` | 48422 | 1026 | LF |
| `rl/breakout/physics_guided_ppo_breakout_v34.py` | `0439ac5248fb9424081ff4cd0c1080ad36bf042f` | 60953 | 1236 | LF |
| `rl/breakout/physics_guided_ppo_breakout_v4.py` | `04598f24a97fc34eefdf8fe9f2ff8a7c9aa46fcd` | 22643 | 527 | LF |
| `rl/breakout/physics_guided_ppo_breakout_v9.py` | `0db4bb5d70eed07e1696f9c1537195456bc875bc` | 24661 | 553 | LF |
| `rl/route_d_vit_policy/physics_rl_core.py` | `7177d83dd1cd3b29c7e95d91ec2ba299e13fe66a` | 10963 | 315 | LF |
| `rl/route_d_vit_policy/train_ppo_vit_breakout_v1.py` | `1d2e91ab520b2febfc2af498a1ef832ab16d1ad5` | 16778 | 387 | LF |
| `rl/route_d_vit_policy/vit_policy_atari.py` | `2a4d0022135ed1a40e37e379fe8493af330a826f` | 9281 | 283 | LF |
| `nn/README.md` | `6b5f49e12181c81662412f984d555c8e98b72659` | 3242 | 74 | LF |
| `nn/cifar10/cnn10_v101.py` | `e01456c300c60735def259d58a221af9c1b991a4` | 282280 | 4863 | LF |
| `nn/cifar10/cnn10_v110.py` | `03ab0a433b394aca94d134c2c8a1d7b91aa17594` | 335382 | 5732 | LF |
| `nn/cifar10/cnn10_v111.py` | `afd532d38074a77a062d0af81725b66191ed9910` | 362603 | 6175 | LF |
| `nn/cifar10/cnn10_v112.py` | `77d894fbb6be9e92e41c021dbf698dc3f8e12cc8` | 365967 | 6261 | LF |
| `nn/cifar10/cnn10_v117.py` | `3a3bf7c8c6db008f2ef5b4db943b43594be3c5ee` | 426586 | 7430 | LF |
| `nn/cifar10/cnn10_v118.py` | `399f5dad764f9097c78205b9bdb90f264d960613` | 444301 | 7753 | LF |
| `nn/cifar10/cnn10_v119.py` | `1fda1af5e8a5b4dc09fb860a98aaed461ef6490c` | 465444 | 8154 | LF |
| `nn/cifar10/cnn10_v51.py` | `3516a363737c1930019ba97f7c2853e67fc0eee2` | 167223 | 2985 | LF |
| `nn/cifar100/vit100_v103.py` | `62c1a1a0312cf200fd80e7033afef0d2c775bc55` | 526520 | 8533 | CRLF |
| `nn/cifar100/vit100_v110.py` | `59a135268542c50e764ae3ba31bbfb2758987879` | 699142 | 11719 | CRLF |
| `nn/cifar100/vit100_v111.py` | `cf269ef5d9c1d0bfad4c8b0c97b8a848777cacc9` | 728288 | 12240 | CRLF |
| `nn/cifar100/vit100_v77.py` | `ebad21c43c1374b8dce5548efb39b86b71443731` | 242428 | 4172 | CRLF |
| `nn/cifar100/vit100_v95.py` | `bc5f77aea41067cbc6b1acabe52f6529695aa240` | 409842 | 6630 | CRLF |
| `nn/cifar10_infodiff/cnn10_info_v01.py` | `7a1fe72c98e8665857422bf5929345206512f7ae` | 44486 | 1079 | LF |
| `nn/cifar10_infodiff/cnn10_info_v02.py` | `8c708d2dcce917c369db732e2a51b07b5782c72a` | 77985 | 1622 | LF |
| `results/README.md` | `f8e22940da381e94d7b985cfef7e275e7cce993b` | 2440 | 43 | LF |
| `results/nn/results_cnn_info_v01_seed1234.json` | `3e62437113b6a20db4278583d90ce6ae134fde66` | 13639 | 1046 | LF |
| `results/nn/results_cnn_info_v02_seed1234.json` | `1db80d71b98151ac742955cb76fc03f92121de54` | 20812 | 1248 | LF |
| `results/nn/results_cnn_v110_seed42.json` | `b562e7501dc6f27447faab4227d41665009cba1f` | 67077 | 2172 | LF |
| `results/nn/results_cnn_v110_seed42_core.json` | `eba52b7bc508d4e9628962b36338a1045c4db31a` | 48801 | 1828 | LF |
| `results/nn/results_cnn_v111_seed42.json` | `5b002a09a3bb6109c04e1dc8e95c1ed3926e4c41` | 77158 | 2232 | LF |
| `results/nn/results_cnn_v111_seed42_core.json` | `3bfed4206895b7b66b46eb34a88f15099961d64a` | 58949 | 1888 | LF |
| `results/nn/results_cnn_v112_seed42.json` | `8244536a5f68d260dce8e7b5c09986d08cb79d39` | 76510 | 2313 | LF |
| `results/nn/results_cnn_v112_seed42_core.json` | `efe5c0e2f9d73e19a48a71e3fefc610edbc68757` | 58233 | 1969 | LF |
| `results/nn/results_cnn_v117_seed42.json` | `e0ba0c2c3e20fb4e19d1422ca14ab2bb70caca8b` | 96619 | 2586 | LF |
| `results/nn/results_cnn_v117_seed42_core.json` | `cf73ae12503733abeb8131e56bd4acf2c8eed985` | 78347 | 2242 | LF |
| `results/nn/results_cnn_v118_seed42.json` | `f8b7611464cf87435784276973721cbd688414e4` | 103988 | 2893 | LF |
| `results/nn/results_cnn_v118_seed42_core.json` | `e99e3e1b68caec36a68852ea2d0465d2ab2c819e` | 85714 | 2549 | LF |
| `results/nn/results_cnn_v51_seed1234.json` | `c18e40d3794d39fbee016b48c4ad98586ca8340a` | 27291 | 1474 | LF |
| `results/rl/env_check.txt` | `04f2f790f8443447b6b1a77bfa5a37c49fbfcd99` | 100 | 4 | MIXED |
| `results/rl/verify_ppo_v10_out.txt` | `edb54480598190c769d7e2eb42ac0c2643b4f558` | 12900 | 134 | LF |
| `results/rl/verify_ppo_v21_out.txt` | `f4b6aa7ae86ad11023d91c0049edca65dde5d302` | 11082 | 230 | CRLF |
| `results/rl/verify_ppo_v22_out.txt` | `bf0697ebfc9d2566fff2d6ee59e87a54f8e5a8dd` | 13812 | 158 | LF |
| `results/rl/verify_ppo_v9_out.txt` | `57b344fc4db860dcf262dc6e6e8ad6b120c28330` | 10128 | 113 | LF |
| `tools/README.md` | `ddcd152fd79905c77f6f0b5271e86b924f433477` | 3250 | 62 | LF |
| `tools/lint_train_script.py` | `2fc57caf6899fcba01844c0b1faef130d2107e3c` | 28803 | 578 | CRLF |
| `tools/make_manifest.py` | `5a2d602934b3ac26cf73098a48424f8f00c8546a` | 3143 | 97 | LF |
| `tools/selfcheck/_cnn10_v119_exec_selftest.py` | `e6e1aca2eb3f9c4c2d981ac333c0806a6634cd90` | 28719 | 562 | CRLF |
| `tools/selfcheck/_deploy_check_cnn10_v119.py` | `bb99271e9cec6bbd2160ca07e8ab3b02fcc988ec` | 15226 | 309 | CRLF |
| `tools/selfcheck/_gen_cnn10_v119.py` | `58e68694f4a7b034c43c6861efe6388ff1cf592f` | 33217 | 647 | CRLF |
| `tools/selfcheck/_v119_launch_path_check.py` | `d85a28815e785b910e51a597b81224d9e8036edf` | 7593 | 189 | CRLF |

总计 **56** 个文件。
