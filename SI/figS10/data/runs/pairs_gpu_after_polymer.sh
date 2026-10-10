#!/bin/bash
# 2026-09-30: the GPU arm of the reaction-count sweep, started only once the fcc_si polymer campaign has
# released GPU 7 (the one GPU this work may use), so neither timing shares the device.
until grep -q FCC_SI_CAMPAIGN_COMPLETE /raid/racda/lmrddme_polymer/fcc_si/campaign.out; do sleep 30; done
cd /home/racda/ftsz_tests/studies/lmrddme_validation/advantage
ADV_SCHEMES=pr0,pr1,pr2,pr3,pr4,pr6 ADV_ETAS=0.09 ADV_ARMS=lat_gpu ADV_GPU=7 \
  /raid/racda/opt/readdy_env/bin/python adv_campaign.py
