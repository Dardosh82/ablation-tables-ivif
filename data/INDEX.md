# Data index

Every recorded result of the experiments in the paper: per-run metric files (per-image rows where the
evaluator wrote them), by the folder they were written to. Run numbers are those of the paper and of
`registrations/`, which holds each confirmatory run's pre-registration, written before the run.

## Pre-registrations

- `registrations/CrossFuse/Tier2_CrossFuse/RUN31_PREREGISTRATION.md`: Run 31 — pre-registration · case study I's σ from three seeds to eight
- `registrations/CrossFuse/Tier3_CrossFuse/RUN11_PREREGISTRATION.md`: Run 11 — pre-registration · case study I's two three-seed arms, taken to five
- `registrations/CrossFuse/Tier3_CrossFuse/RUN28_PREREGISTRATION.md`: Run 28 (= C2): the volume-controlled Top-K test on CrossFuse itself. Pre-registration
- `registrations/CrossFuse/Tier3_CrossFuse/RUN30_PREREGISTRATION.md`: Run 30 — pre-registration · does removing the decoder's skip remove the MI/VIF inversion?
- `registrations/LRRNet/PERSON_FOLLOWUP_PREREGISTRATION.md`: Person-contrast follow-up: pre-registration (seeds 43 and 44)
- `registrations/LRRNet/RUN10_PREREGISTRATION.md`: Run 10 — pre-registration
- `registrations/LRRNet/RUN12_PREREGISTRATION.md`: Run 12 — pre-registration · M1 (weight tying) from three seeds to five
- `registrations/LRRNet/RUN13_PREREGISTRATION.md`: Run 13 — pre-registration · the training-length curve: does the tied network have headroom the published one lacks?
- `registrations/LRRNet/RUN14_PREREGISTRATION.md`: Run 14 — pre-registration · free depth: the tied network at depth 8, from three seeds to five
- `registrations/LRRNet/RUN16_PREREGISTRATION.md`: Run 16 — pre-registration · the LRRNet verdicts on a second test set (RoadScene), no training
- `registrations/LRRNet/RUN17_PREREGISTRATION.md`: Run 17 — pre-registration · the one open lead: tied against untied at depth 8, on new seeds
- `registrations/LRRNet/RUN17B_PREREGISTRATION.md`: Run 17b — pre-registration · Run 17's checkpoints on a second test set (RoadScene), no training
- `registrations/LRRNet/RUN18_PREREGISTRATION.md`: Run 18 — pre-registration · the improvement screen (S0–S4), exploratory by design
- `registrations/LRRNet/RUN19_PREREGISTRATION.md`: Run 19 — pre-registration · confirming the arm the Run 18 screen selects
- `registrations/LRRNet/RUN20_PREREGISTRATION.md`: Run 20 — pre-registration · a general test of the tied depth-8 network against the published configuration
- `registrations/LRRNet/RUN21_PREREGISTRATION.md`: Run 21 — pre-registration · a third test set (MSRS, 361 pairs) for the depth-8 result, no training
- `registrations/LRRNet/RUN22_PREREGISTRATION.md`: Run 22 — pre-registration · Run 18's collaborator-account arms, again on our own account
- `registrations/LRRNet/RUN23_PREREGISTRATION.md`: Run 23 — pre-registration · a screen for removing the tied depth-8 network's SD cost, exploratory by design
- `registrations/LRRNet/RUN23B_PREREGISTRATION.md`: Run 23b — pre-registration · one more screening arm, aimed at where diagnostic D4 found the SD cost
- `registrations/LRRNet/RUN24_PREREGISTRATION.md`: Run 24 — pre-registration · confirmation of Run 23's candidate: an improvement without a cost?
- `registrations/LRRNet/RUN25_PREREGISTRATION.md`: Run 25: the positive control at twenty seeds per arm. Pre-registration
- `registrations/LRRNet/RUN26_PREREGISTRATION.md`: Run 26: can deterministic mode make a seed sufficient? Pre-registration
- `registrations/LRRNet/RUN27_PREREGISTRATION.md`: Run 27: the volume-controlled Top-K test at eight seeds per arm. Pre-registration
- `registrations/LRRNet/RUN29_PREREGISTRATION.md`: Run 29 — pre-registration · do the headline results hold over test images as well as over runs?
- `registrations/LRRNet/RUN32_PREREGISTRATION.md`: Run 32 — pre-registration · do the metric verdicts agree with a downstream task?
- `registrations/LRRNet/RUN9_PREREGISTRATION.md`: Run 9 — pre-registration
- `registrations/NoSkipControl/PREREGISTRATION.md`: Run 15 — pre-registration · does the MI/VIF inversion follow the input skip? Two published methods without one

## Result folders

| Folder | Files |
|---|---|
| `data/CDDFuse/kernel_mechanism` | 1 |
| `data/CDDFuse/results_mechanism/kaggle/results_cddfuse_mechanism` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/kernel_run31_a` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/kernel_run31_b` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/kernel_run31_c` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/results_perimage/tables_R1_raw` | 6 |
| `data/CrossFuse/Tier2_CrossFuse/results_perimage/tables_shipped_raw` | 6 |
| `data/CrossFuse/Tier2_CrossFuse/results_R1/CrossFuse/models` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/results_R1b/CrossFuse/models` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/results_R1b/tables_R1b_raw` | 6 |
| `data/CrossFuse/Tier2_CrossFuse/results_R2/models` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/results_R2/tables/tables_R2_raw` | 6 |
| `data/CrossFuse/Tier2_CrossFuse/results_R2/tables/tables_R2_rgbluma` | 6 |
| `data/CrossFuse/Tier2_CrossFuse/results_R3/CrossFuse/models` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/results_R3/tables_R3_raw` | 6 |
| `data/CrossFuse/Tier2_CrossFuse/results_R3/tables_R3_rgbluma` | 6 |
| `data/CrossFuse/Tier2_CrossFuse/results_R3b/CrossFuse/models` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/results_R3b/tables_R3b_raw` | 6 |
| `data/CrossFuse/Tier2_CrossFuse/results_R3c/CrossFuse/models` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/results_R3c/tables_R3c_raw` | 6 |
| `data/CrossFuse/Tier2_CrossFuse/results_R4/CrossFuse/models` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/results_run31/cf-run31-a` | 2 |
| `data/CrossFuse/Tier2_CrossFuse/results_run31/cf-run31-a/models_R3s45` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/results_run31/cf-run31-a/models_R3s46` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/results_run31/cf-run31-a/tables_R3s45_raw` | 6 |
| `data/CrossFuse/Tier2_CrossFuse/results_run31/cf-run31-a/tables_R3s46_raw` | 6 |
| `data/CrossFuse/Tier2_CrossFuse/results_run31/cf-run31-b` | 2 |
| `data/CrossFuse/Tier2_CrossFuse/results_run31/cf-run31-b/models_R3s47` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/results_run31/cf-run31-b/models_R3s48` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/results_run31/cf-run31-b/tables_R3s47_raw` | 6 |
| `data/CrossFuse/Tier2_CrossFuse/results_run31/cf-run31-b/tables_R3s48_raw` | 6 |
| `data/CrossFuse/Tier2_CrossFuse/results_run31/cf-run31-c` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/results_run31/cf-run31-c/models_R3s49` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/results_run31/cf-run31-c/tables_R3s49_raw` | 6 |
| `data/CrossFuse/Tier2_CrossFuse/results_run31` | 2 |
| `data/CrossFuse/Tier2_CrossFuse/upload_code` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/upload_models` | 1 |
| `data/CrossFuse/Tier2_CrossFuse/upload_r2model` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/.templates/tier3-a1d` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/kernel_crossfuse-run30-noskip` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/kernel_halfbudget_cf/.template` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/kernel_halfbudget_cf` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/kernel_halfbudget_cf_b/.template` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/kernel_halfbudget_cf_b` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/kernel_mechanism_cf` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/kernel_poscontrol_cf/.template` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/kernel_poscontrol_cf` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/kernel_run11_hb` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/kernel_run11_hb_score` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/kernel_run11_k0` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/kernel_run28_corpus` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/kernel_run28_s47` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/kernel_run28_s48` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/kernel_run28_s49` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/kernel_run28_s50` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/kernel_run28_s51` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_A1a/CrossFuse/models` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_A1a/tables_A1a_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_A1b/CrossFuse/models` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_A1b/tables_A1b_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_A1c/CrossFuse/models` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_A1c/tables_A1c_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_A1d/CrossFuse/models` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_A1d/tables_A1d_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_A2/CrossFuse/models` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_A2/tables_A2_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_A3/CrossFuse/models` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_A3/tables_A3_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_A4/CrossFuse/models` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_A4/tables_A4_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_A5/CrossFuse/models` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_A5/tables_A5_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_A/CrossFuse/models` | 3 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_A/CrossFuse/models_HB42` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_A/CrossFuse/models_HB43` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_A/CrossFuse/models_HB44` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_A/tables_HB42_raw` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_A/tables_HB43_raw` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_A/tables_HB44_raw` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_B/CrossFuse/models` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_B/CrossFuse/models_HB43` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_B/CrossFuse/models_HB44` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_B/tables_HB43_raw` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_B/tables_HB44_raw` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_cf` | 8 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_cf/tables_A1b_raw` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_cf/tables_A1c_raw` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_cf/tables_A1d_raw` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_cf/tables_HB42_raw` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_cf/tables_HB43_raw` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_cf/tables_HB44_raw` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_cf/tables_PC45_raw` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_halfbudget_cf/tables_PC46_raw` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_mechanism_cf/kaggle` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_mechanism_cf/kaggle/tables/PC45` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_mechanism_cf/kaggle/tables/PC46` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_mechanism_cf/kaggle/tables/U42` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_mechanism_cf/kaggle/tables/U43` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_mechanism_cf/kaggle/tables/U44` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_mechanism_cf/kaggle/tables/U45` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_mechanism_cf/kaggle/tables/U46` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf` | 5 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf/tables_A1b_raw` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf/tables_A1c_raw` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf/tables_A1d_raw` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf/tables_PC45_raw` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf/tables_PC46_raw` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf/tables_U42_raw` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf/tables_U43_raw` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf/tables_U44_raw` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf/tables_U45_raw` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf/tables_U46_raw` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf_rawdownload/CrossFuse/models` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf_rawdownload/CrossFuse/models_PC46` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf_rawdownload/tables_PC45_raw` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf_rawdownload/tables_PC46_raw` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf_rawdownload/tables_U42_raw` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf_rawdownload/tables_U43_raw` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf_rawdownload/tables_U44_raw` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf_rawdownload/tables_U45_raw` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_poscontrol_cf_rawdownload/tables_U46_raw` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run11_hb` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run11_hb/tables_HB45_raw` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run11_hb/tables_HB46_raw` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run11_k0` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run11_k0/tables_A1e_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_run11_k0/tables_A1f_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_run28_corpus` | 1 |
| `data/CrossFuse/Tier3_CrossFuse/results_run28_s47` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run28_s47/tables_RD47_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_run28_s47/tables_TK47_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_run28_s48` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run28_s48/tables_RD48_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_run28_s48/tables_TK48_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_run28_s49` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run28_s49/tables_RD49_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_run28_s49/tables_TK49_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_run28_s50` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run28_s50/tables_RD50_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_run28_s50/tables_TK50_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_run28_s51` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run28_s51/tables_RD51_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_run28_s51/tables_TK51_raw` | 6 |
| `data/CrossFuse/Tier3_CrossFuse/results_run30` | 3 |
| `data/CrossFuse/Tier3_CrossFuse/results_run30/tables/U42_OFF` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run30/tables/U42_ON` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run30/tables/U43_OFF` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run30/tables/U43_ON` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run30/tables/U44_OFF` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run30/tables/U44_ON` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run30/tables/U45_OFF` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run30/tables/U45_ON` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run30/tables/U46_OFF` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/results_run30/tables/U46_ON` | 2 |
| `data/CrossFuse/Tier3_CrossFuse` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/upload_k0` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/upload_poscontrol` | 2 |
| `data/CrossFuse/Tier3_CrossFuse/upload_run32_ckpts` | 1 |
| `data/LRRNet/.templates/lrrnet-m1-nb8` | 1 |
| `data/LRRNet/.templates/lrrnet-m1-tied` | 1 |
| `data/LRRNet/.templates/lrrnet-stage1` | 1 |
| `data/LRRNet/.templates/lrrnet-stage2` | 1 |
| `data/LRRNet/.templates/lrrnet-stage3` | 1 |
| `data/LRRNet/.templates/lrrnet-stage4` | 1 |
| `data/LRRNet` | 3 |
| `data/LRRNet/kernel_csv/lrrnet-m1-nb8` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-m1-tied` | 2 |
| `data/LRRNet/kernel_csv/lrrnet-m2` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run12-m1tied` | 2 |
| `data/LRRNet/kernel_csv/lrrnet-run13-tied-a` | 2 |
| `data/LRRNet/kernel_csv/lrrnet-run13-tied-b` | 2 |
| `data/LRRNet/kernel_csv/lrrnet-run13-untied-a` | 2 |
| `data/LRRNet/kernel_csv/lrrnet-run13-untied-b` | 2 |
| `data/LRRNet/kernel_csv/lrrnet-run14-m1tied-nb8` | 2 |
| `data/LRRNet/kernel_csv/lrrnet-run17-tied-a` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run17-tied-b` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run17-tied-c` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run17-untied-a` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run17-untied-b` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run17-untied-c` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run18-s1a` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run18-s1b` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run18-s2` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run18-s3t` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run18-s3u` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run18-s4` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run20-tied-a` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run20-tied-b` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run20-tied-c` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run22-s1a` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run22-s2` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run22-s4` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run23-a1` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run23-a2` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run23-a3` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run23-a4` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-run23b-b1` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-stage1` | 2 |
| `data/LRRNet/kernel_csv/lrrnet-stage2` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-stage3` | 1 |
| `data/LRRNet/kernel_csv/lrrnet-stage4` | 1 |
| `data/LRRNet/kernel_m3_k1000_src` | 1 |
| `data/LRRNet/kernel_run10_nb2` | 1 |
| `data/LRRNet/kernel_run10_nb4a` | 1 |
| `data/LRRNet/kernel_run10_nb4b` | 1 |
| `data/LRRNet/kernel_run10_nb6a` | 1 |
| `data/LRRNet/kernel_run10_nb6b` | 1 |
| `data/LRRNet/kernel_run12_m1tied` | 1 |
| `data/LRRNet/kernel_run13_tied_a` | 1 |
| `data/LRRNet/kernel_run13_tied_b` | 1 |
| `data/LRRNet/kernel_run13_untied_a` | 1 |
| `data/LRRNet/kernel_run13_untied_b` | 1 |
| `data/LRRNet/kernel_run14_m1tied_nb8` | 1 |
| `data/LRRNet/kernel_run16_fuse` | 1 |
| `data/LRRNet/kernel_run16_score` | 1 |
| `data/LRRNet/kernel_run17_tied_a` | 1 |
| `data/LRRNet/kernel_run17_tied_b` | 1 |
| `data/LRRNet/kernel_run17_tied_c` | 1 |
| `data/LRRNet/kernel_run17_untied_a` | 1 |
| `data/LRRNet/kernel_run17_untied_b` | 1 |
| `data/LRRNet/kernel_run17_untied_c` | 1 |
| `data/LRRNet/kernel_run17b_fuse` | 1 |
| `data/LRRNet/kernel_run17b_score` | 1 |
| `data/LRRNet/kernel_run18_s0` | 1 |
| `data/LRRNet/kernel_run18_s1a` | 1 |
| `data/LRRNet/kernel_run18_s1b` | 1 |
| `data/LRRNet/kernel_run18_s2` | 1 |
| `data/LRRNet/kernel_run18_s3t` | 1 |
| `data/LRRNet/kernel_run18_s3u` | 1 |
| `data/LRRNet/kernel_run18_s4` | 1 |
| `data/LRRNet/kernel_run20_cpu_a` | 1 |
| `data/LRRNet/kernel_run20_cpu_b` | 1 |
| `data/LRRNet/kernel_run20_cpu_c` | 1 |
| `data/LRRNet/kernel_run20_fuse` | 1 |
| `data/LRRNet/kernel_run20_score` | 1 |
| `data/LRRNet/kernel_run20_tied_a` | 1 |
| `data/LRRNet/kernel_run20_tied_b` | 1 |
| `data/LRRNet/kernel_run20_tied_c` | 1 |
| `data/LRRNet/kernel_run21_fuse` | 1 |
| `data/LRRNet/kernel_run21_score` | 1 |
| `data/LRRNet/kernel_run22_s1a` | 1 |
| `data/LRRNet/kernel_run22_s2` | 1 |
| `data/LRRNet/kernel_run22_s4` | 1 |
| `data/LRRNet/kernel_run23_a1` | 1 |
| `data/LRRNet/kernel_run23_a2` | 1 |
| `data/LRRNet/kernel_run23_a3` | 1 |
| `data/LRRNet/kernel_run23_a4` | 1 |
| `data/LRRNet/kernel_run23b_b1` | 1 |
| `data/LRRNet/kernel_run25_fuse` | 1 |
| `data/LRRNet/kernel_run25_score` | 1 |
| `data/LRRNet/kernel_run26_det` | 1 |
| `data/LRRNet/kernel_run27_k0` | 1 |
| `data/LRRNet/kernel_run27_topk` | 1 |
| `data/LRRNet/kernel_run29_score_src16` | 1 |
| `data/LRRNet/kernel_run29_score_src17b` | 1 |
| `data/LRRNet/kernel_run29_score_src20` | 1 |
| `data/LRRNet/kernel_run29_score_src25` | 1 |
| `data/LRRNet/kernel_run32_detect` | 1 |
| `data/LRRNet/kernel_run9_nb2` | 1 |
| `data/LRRNet/kernel_run9_nb4` | 1 |
| `data/LRRNet/kernel_run9_nb6` | 1 |
| `data/LRRNet/kernel_run9_nb8` | 1 |
| `data/LRRNet/results_lrrnet-run22-s1a` | 1 |
| `data/LRRNet/results_lrrnet-run22-s2` | 1 |
| `data/LRRNet/results_lrrnet-run22-s4` | 1 |
| `data/LRRNet/results_lrrnet-run23-a1` | 1 |
| `data/LRRNet/results_lrrnet-run23-a2` | 1 |
| `data/LRRNet/results_lrrnet-run23-a3` | 1 |
| `data/LRRNet/results_lrrnet-run23-a4` | 1 |
| `data/LRRNet/results_lrrnet-run23b-b1` | 1 |
| `data/LRRNet/results_m3_k1000` | 1 |
| `data/LRRNet/results_run10_nb2` | 1 |
| `data/LRRNet/results_run10_nb4a` | 1 |
| `data/LRRNet/results_run10_nb4b` | 1 |
| `data/LRRNet/results_run10_nb6a` | 1 |
| `data/LRRNet/results_run10_nb6b` | 1 |
| `data/LRRNet/results_run16` | 2 |
| `data/LRRNet/results_run17b` | 2 |
| `data/LRRNet/results_run18_s0` | 1 |
| `data/LRRNet/results_run20/cpu_a` | 10 |
| `data/LRRNet/results_run20/cpu_b` | 10 |
| `data/LRRNet/results_run20/cpu_c` | 10 |
| `data/LRRNet/results_run20` | 2 |
| `data/LRRNet/results_run20_gpu_score` | 2 |
| `data/LRRNet/results_run21` | 2 |
| `data/LRRNet/results_run25` | 1 |
| `data/LRRNet/results_run26` | 1 |
| `data/LRRNet/results_run27_k0` | 1 |
| `data/LRRNet/results_run27_topk` | 1 |
| `data/LRRNet/results_run29` | 15 |
| `data/LRRNet/results_run32/run32` | 4 |
| `data/LRRNet/results_run32/run32/preds` | 37 |
| `data/LRRNet/results_run32` | 1 |
| `data/LRRNet/results_run9_nb2` | 1 |
| `data/LRRNet/results_run9_nb4` | 1 |
| `data/LRRNet/results_run9_nb6` | 1 |
| `data/LRRNet/results_run9_nb8` | 1 |
| `data/LRRNet/upload_scripts` | 1 |
| `data/NoSkipControl/results` | 2 |
