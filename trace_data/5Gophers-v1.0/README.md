## OVERVIEW

5Gophers 1.0 is a dataset collected when the world's very first commercial 5G services were made available to consumers. It should serve as a baseline to evaluate the 5G's performance evolution over time. Results using this dataset is presented in our measurement paper - "A First Look at Commercial 5G Performance on Smartphones".

This dataset is being made available to the research community.

## FILES and FOLDER STRUCTURE

All the files are in CSV format with headers that should hopefully be self-explainatory.

5Gophers-v1.0
├── All-Carriers 
│   ├── 01-Throughput
│   ├── 02-Round-Trip-Times
│   └── 03-User-Mobility
└── mmWave-only
    ├── 03-UE-Panel (LoS Tests)
    ├── 04-Ping-Traces (Latency Tests)
    ├── 05-UE-Panel (NLoS Tests)
    ├── 06-UE-Panel (Orientation Tests)
    ├── 07-UE-Panel (Distance Tests)
    ├── 08-Web-Page-Load-Tests
    ├── 09-HTTPS-CDN-vs-NonCDN (Download Test)
    └── 10-HTTP-vs-HTTPS (Download Test)

## CITING THE DATASET

```
@inproceedings{10.1145/3366423.3380169,
  author = {Narayanan, Arvind and Ramadan, Eman and Carpenter, Jason and Liu, Qingxu and Liu, Yu and Qian, Feng and Zhang, Zhi-Li},
  title = {A First Look at Commercial 5G Performance on Smartphones},
  year = {2020},
  isbn = {9781450370233},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3366423.3380169},
  doi = {10.1145/3366423.3380169},
  booktitle = {Proceedings of The Web Conference 2020},
  pages = {894–905},
  numpages = {12},
  location = {Taipei, Taiwan},
  series = {WWW ’20}
}
```

## QUESTIONS?

Please feel free to contact the FiveGophers team for information about the data (fivegophers@umn.edu, naray111@umn.edu)

## LICENSE 

5Gophers 1.0 dataset is licensed under the Creative Commons Attribution 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.