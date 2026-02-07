Expects `../EFTA00400459-{000..=075}_2x.png` to exist

* Run `./train.sh` to generate training from train_top.txt and train_bot.txt corresponding to page-001_2x.png
* Run `./run.sh` to OCR all pages and generated recovered.pdf

Trains from top of page-001 and bottom of page-001 non-contiguously to capture vertical drift.
Memorizes grid location and reuses for subsequent pages (non-training runs) to prevent pixel shifts.
