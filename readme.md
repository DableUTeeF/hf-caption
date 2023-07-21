# Table
| id     | desc, `layer`x`head`              | rouge1  | bleu | features     | batch |
|--------|-----------------------------------|---------|------|--------------|-------|
| 318129 | baseline                          | 42.7588 | 1    | (50, 768)    | 16    |
| 318159 | dino, 8x8                         | 40.9833 | 1    | (900, 256)   | 16    |
| 318179 | dino, 2x2                         | 41.0165 | 1    | (900, 256)   | 16    |
| 361808 | baseline, sd                      | 1       | 1    | (50, 768)    | 16    |
| 368249 | dino2x2, 800                      | 40.4109 | 1    | (900, 256)   | 16    |
| 368253 | dino, 800, cached (not finished)  | 41.0445 | 1    | (900, 256)   | 16    |
| 368252 | rcnn, 1333, cached (not finished) | 41.532  | 1    | (1000, 1024) | 16    |
| 368254 | dino, 1333, cached (not finished) | 40.904  | 1    | (900, 256)   | 16    |
| 370558 | dino, 800, cached                 |         | 1    | (900, 256)   | 16    |
| 370557 | rcnn, 1333, cached                |         | 1    | (1000, 1024) | 16    |
| 370559 | dino, 1333, cached                |         | 1    | (900, 256)   | 16    |
| 371559 | dino, 800, cached , 50 queries    |         | 1    | (50, 256)    | 16    |
