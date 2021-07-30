## OVERVIEW

Lumos5G 1.0 is a dataset that represents the `Loop` area of the IMC'20 paper - "Lumos5G: Mapping and Predicting Commercial mmWave 5G Throughput". The Loop area is a 1300 meter loop near U.S. Bank Stadium in Minneapolis downtown area that covers roads, railroad crossings, restaurants, coffee shops, and recreational outdoor parks. 

This dataset is being made available to the research community.

## DATASET COLUMNS AND DESCRIPTION

The description of the columns in the dataset CSV, from left to right, are:

- `run_num`: Indicates the run number. For each trajectory and mobility mode, we conduct several runs of experiments. 
- `seq_num`: This is the sequence number. For each run, the sequence number acts like an index or a per-second timeline.
- `abstractSignalStr`: Indicates the abstract signal strength as reported by Android API (https://developer.android.com/reference/android/telephony/SignalStrength#getLevel()). No matter whether the UE was connected to 5G service or not, this column always reported a value associated with the LTE/4G radio. Note, if one is interested to understand the signal strength values related to 5G-NR, we refer them to other columns such as `nr_ssRsrp`, `nr_ssRsrq`, and `nr_ssSinr`.
- `latitude`: The latitude in degrees as reported by Android's API (https://developer.android.com/reference/android/location/Location#getLatitude()).
- `longitude`: The longitude in degrees as reported by Android's API (https://developer.android.com/reference/android/location/Location#getLongitude()).
- `movingSpeed`: The ground mobility/moving speed of the UE as reported by Android's API (https://developer.android.com/reference/android/location/Location#getSpeed()). The unit is meters per second.
- `compassDirection`: The bearing in degrees as reported by Android's API (https://developer.android.com/reference/android/location/Location#getBearing()). Bearing is the horizontal direction of travel of this device, and is not related to the device orientation. It is guaranteed to be in the range `(0.0, 360.0]` if the device has a bearing. 
- `nrStatus`: Indicates if the UE was connected to 5G network or not. When `nrStatus=CONNECTED`, the UE was connected to 5G. All other values of `nrStatus` such as `NOT_RESTRICTED` and `NONE` indicate the UE was not connected to 5G. `nrStatus` was obtained by parsing the raw string representation of `ServiceState` object (https://developer.android.com/reference/android/telephony/ServiceState#toString()). 
- `lte_rssi`: Get Received Signal Strength Indication (RSSI) in dBm of the primary serving LTE cell. The value range is [-113, -51] inclusively or CellInfo#UNAVAILABLE if unavailable. Reference: TS 27.007 8.5 Signal quality +CSQ.
- `lte_rsrp`: Get reference signal received power (RSRP) in dBm of the primary serving LTE cell.
- `lte_rsrq`: Get reference signal received quality (RSRQ) of the primary serving LTE cell.
- `lte_rssnr`: Get reference signal signal-to-noise ratio (RSSNR) of the primary serving LTE cell.
- `nr_ssRsrp`: Obtained by parsing the raw string representation of `SignalStrength` object (https://developer.android.com/reference/android/telephony/SignalStrength#toString()). `nr_ssRsrp` was a field in this object's `CellSignalStrengthNr` section. In general, this value was only available when the UE was connected to 5G (i.e., when `nrStatus=CONNECTED`). Reference: 3GPP TS 38.215. Range: -140 dBm to -44 dBm.
- `nr_ssRsrq`: Obtained by parsing the raw string representation of `SignalStrength` object (https://developer.android.com/reference/android/telephony/SignalStrength#toString()). `nr_ssRsrq` was a field in this object's `CellSignalStrengthNr` section. In general, this value was only available when the UE was connected to 5G (i.e., when `nrStatus=CONNECTED`). Reference: 3GPP TS 38.215. Range: -20 dB to -3 dB.
- `nr_ssSinr`: Obtained by parsing the raw string representation of `SignalStrength` object (https://developer.android.com/reference/android/telephony/SignalStrength#toString()). `nr_ssSinr` was a field in this object's `CellSignalStrengthNr` section. In general, this value was only available when the UE was connected to 5G (i.e., when `nrStatus=CONNECTED`). Reference: 3GPP TS 38.215 Sec 5.1.*, 3GPP TS 38.133 10.1.16.1 Range: -23 dB to 40 dB
- `Throughput`: Indicates the throughput perceived by the UE. iPerf 3.7 was used to measure the per-second TCP downlink at the UE.
- `mobility_mode`: Indicates the grouth truth about the mobility mode when the experiment was conducted. This value can either be walking or driving.
- `trajectory_direction`: Indicates the ground truth about the trajectory direction of the experiment conducted at the Loop area. `CW` indicates clockwise direction, while `ACW` indicates anti-clockwise. Note, the driving experiments were only conducted in `CW` direction as certain parts of the loop were one way only. Walking-based experiments were conducted in both directions. 
- `tower_id`: Indicates the (anonymized) tower identifier.

Note: We found that availability (and at times even the values) of `lte_rssi`, `nr_ssRsrp`, `nr_ssRsrq` and `nr_ssSinr` were not reliable. Since these values were sampled every second, at certain times (e.g., boundary cases), we might still find NR-related values when `nrStatus` is not equal to `CONNECTED`. However, in this dataset, we still include all the raw values as reported by the APIs. 


## CITING THE DATASET

```
@inproceedings{10.1145/3419394.3423629,
  author = {Narayanan, Arvind and Ramadan, Eman and Mehta, Rishabh and Hu, Xinyue and Liu, Qingxu and Fezeu, Rostand A. K. and Dayalan, Udhaya Kumar and Verma, Saurabh and Ji, Peiqi and Li, Tao and Qian, Feng and Zhang, Zhi-Li},
  title = {Lumos5G: Mapping and Predicting Commercial MmWave 5G Throughput},
  year = {2020},
  isbn = {9781450381383},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3419394.3423629},
  doi = {10.1145/3419394.3423629},
  booktitle = {Proceedings of the ACM Internet Measurement Conference},
  pages = {176–193},
  numpages = {18},
  keywords = {bandwidth estimation, mmWave, machine learning, Lumos5G, throughput prediction, deep learning, prediction, 5G},
  location = {Virtual Event, USA},
  series = {IMC '20}
}
```

## QUESTIONS?

Please feel free to contact the FiveGophers/Lumos5G team for questions or information about the data (arvind@cs.umn.edu,eman@cs.umn.edu,zhzhang@cs.umn.edu,fengqian@umn.edu,fivegophers@umn.edu)

## LICENSE 

Lumos5G 1.0 dataset is licensed under the Creative Commons Attribution 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.