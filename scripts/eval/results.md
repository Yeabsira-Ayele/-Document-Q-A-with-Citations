# Chunking comparison

k=5, distance threshold=0.75, 25 questions, 0 with gold pages.

| Strategy | Hit@k | MRR | Page recall | Answered (should) | Refused (should) | Mean best dist. answerable | unanswerable |
|---|---|---|---|---|---|---|---|
| recursive | n/a | n/a | n/a | 1.0 | 0.6 | 0.545 | 0.801 |
| section | n/a | n/a | n/a | 1.0 | 0.6 | 0.516 | 0.771 |

## Per question

### recursive

| id | answerable | best distance | passed gate | top hit |
|---|---|---|---|---|
| q01 | True | 0.4392 | True | YEAB_Technologies_Employee_Handbook.pdf p.12 |
| q02 | True | 0.5287 | True | YEAB_Technologies_Employee_Handbook.pdf p.12 |
| q03 | True | 0.5825 | True | YEAB_Technologies_Employee_Handbook.pdf p.26 |
| q04 | True | 0.5272 | True | YEAB_Technologies_Employee_Handbook.pdf p.21 |
| q05 | True | 0.5499 | True | YEAB_Technologies_Employee_Handbook.pdf p.21 |
| q06 | True | 0.5726 | True | YEAB_Technologies_Employee_Handbook.pdf p.20 |
| q07 | True | 0.4684 | True | YEAB_Technologies_Employee_Handbook.pdf p.42 |
| q08 | True | 0.4585 | True | YEAB_Technologies_Employee_Handbook.pdf p.12 |
| q09 | True | 0.4577 | True | YEAB_Technologies_Employee_Handbook.pdf p.41 |
| q10 | True | 0.5016 | True | YEAB_Technologies_Employee_Handbook.pdf p.41 |
| q11 | True | 0.4865 | True | YEAB_Technologies_Employee_Handbook.pdf p.51 |
| q12 | True | 0.5559 | True | YEAB_Technologies_Employee_Handbook.pdf p.18 |
| q13 | True | 0.7395 | True | YEAB_Technologies_Employee_Handbook.pdf p.2 |
| q14 | True | 0.5106 | True | YEAB_Technologies_Employee_Handbook.pdf p.38 |
| q15 | True | 0.546 | True | YEAB_Technologies_Employee_Handbook.pdf p.10 |
| q16 | True | 0.6254 | True | YEAB_Technologies_Employee_Handbook.pdf p.31 |
| q17 | True | 0.722 | True | YEAB_Technologies_Employee_Handbook.pdf p.20 |
| q18 | True | 0.4287 | True | YEAB_Technologies_Employee_Handbook.pdf p.44 |
| q19 | True | 0.6667 | True | YEAB_Technologies_Employee_Handbook.pdf p.31 |
| q20 | True | 0.5234 | True | YEAB_Technologies_Employee_Handbook.pdf p.25 |
| u01 | False | 0.7212 | True | YEAB_Technologies_Employee_Handbook.pdf p.30 |
| u02 | False | 0.8226 | False | YEAB_Technologies_Employee_Handbook.pdf p.43 |
| u03 | False | 0.6921 | True | YEAB_Technologies_Employee_Handbook.pdf p.4 |
| u04 | False | 0.8837 | False | YEAB_Technologies_Employee_Handbook.pdf p.6 |
| u05 | False | 0.8838 | False | YEAB_Technologies_Employee_Handbook.pdf p.25 |
### section

| id | answerable | best distance | passed gate | top hit |
|---|---|---|---|---|
| q01 | True | 0.3717 | True | YEAB_Technologies_Employee_Handbook.pdf p.12 |
| q02 | True | 0.5344 | True | YEAB_Technologies_Employee_Handbook.pdf p.48 |
| q03 | True | 0.3978 | True | YEAB_Technologies_Employee_Handbook.pdf p.26 |
| q04 | True | 0.4816 | True | YEAB_Technologies_Employee_Handbook.pdf p.21 |
| q05 | True | 0.5545 | True | YEAB_Technologies_Employee_Handbook.pdf p.21-22 |
| q06 | True | 0.5616 | True | YEAB_Technologies_Employee_Handbook.pdf p.20 |
| q07 | True | 0.3988 | True | YEAB_Technologies_Employee_Handbook.pdf p.19 |
| q08 | True | 0.3354 | True | YEAB_Technologies_Employee_Handbook.pdf p.12 |
| q09 | True | 0.5569 | True | YEAB_Technologies_Employee_Handbook.pdf p.41 |
| q10 | True | 0.5463 | True | YEAB_Technologies_Employee_Handbook.pdf p.11 |
| q11 | True | 0.4111 | True | YEAB_Technologies_Employee_Handbook.pdf p.12 |
| q12 | True | 0.4886 | True | YEAB_Technologies_Employee_Handbook.pdf p.18 |
| q13 | True | 0.735 | True | YEAB_Technologies_Employee_Handbook.pdf p.16 |
| q14 | True | 0.5604 | True | YEAB_Technologies_Employee_Handbook.pdf p.26 |
| q15 | True | 0.5919 | True | YEAB_Technologies_Employee_Handbook.pdf p.10-11 |
| q16 | True | 0.61 | True | YEAB_Technologies_Employee_Handbook.pdf p.26 |
| q17 | True | 0.6637 | True | YEAB_Technologies_Employee_Handbook.pdf p.20-21 |
| q18 | True | 0.3918 | True | YEAB_Technologies_Employee_Handbook.pdf p.44 |
| q19 | True | 0.6283 | True | YEAB_Technologies_Employee_Handbook.pdf p.47 |
| q20 | True | 0.5098 | True | YEAB_Technologies_Employee_Handbook.pdf p.25 |
| u01 | False | 0.6537 | True | YEAB_Technologies_Employee_Handbook.pdf p.24 |
| u02 | False | 0.8219 | False | YEAB_Technologies_Employee_Handbook.pdf p.23 |
| u03 | False | 0.6786 | True | YEAB_Technologies_Employee_Handbook.pdf p.8-10 |
| u04 | False | 0.8732 | False | YEAB_Technologies_Employee_Handbook.pdf p.39 |
| u05 | False | 0.8276 | False | YEAB_Technologies_Employee_Handbook.pdf p.18 |
