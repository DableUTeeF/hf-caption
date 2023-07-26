# Table
| id     | desc, `layer`x`head`              | rouge1  | bleu    | features     | batch |
|--------|-----------------------------------|---------|---------|--------------|-------|
| 318129 | baseline                          | .427588 |         | (197, 768)   | 16    |
| 318159 | dino, 8x8                         | .409833 |         | (900, 256)   | 16    |
| 318179 | dino, 2x2                         | .410165 |         | (900, 256)   | 16    |
| 361808 | baseline, sd                      |         |         | (197, 768)   | 16    |
| 368249 | dino2x2, 800                      | .404109 |         | (900, 256)   | 16    |
| 368253 | dino, 800, cached (not finished)  | .410445 |         | (900, 256)   | 16    |
| 368252 | rcnn, 1333, cached (not finished) | .41532  |         | (1000, 1024) | 16    |
| 368254 | dino, 1333, cached (not finished) | .40904  |         | (900, 256)   | 16    |
| 370558 | dino, 800, cached                 |         |         | (900, 256)   | 16    |
| 370557 | rcnn, 1333, cached                |         |         | (1000, 1024) | 16    |
| 370559 | dino, 1333, cached                |         |         | (900, 256)   | 16    |
| 371559 | dino, 800, cached , 50 queries    | .386478 | 0.07997 | (50, 256)    | 16    |
| 373257 | ResNet, baseline                  | .414695 | 0.095   | (49, 2048)   | 16    |
| 373420 | baseline-large                    | .429254 |         | (577, 1024)  | 16    |
| 374944 | vit-large + rcnn-1333             |         |         | (1577, 1024) | 16    |
| 374967 | Swin, baseline                    |         |         | (49, 1024)   | 16    |
| 374967 | Focalnet, baseline                |         |         | (49, 1024)   | 16    |
| 374967 | ResNet, baseline2                 |         |         | (49, 2048)   | 16    |
