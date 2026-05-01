# HAI 21.03 Raw File Audit

Status: `ok`

| file | read_from | rows | attack=0 | attack=1 | missing vs union | not in common |
|---|---|---:|---:|---:|---|---|
| train1.csv | .csv | 216001 | 216001 | 0 | [] | [] |
| train2.csv | .csv | 226801 | 226801 | 0 | [] | [] |
| train3.csv | .csv | 478801 | 478801 | 0 | [] | [] |
| test1.csv | .csv | 43201 | 42572 | 629 | [] | [] |
| test2.csv | .csv | 118801 | 115352 | 3449 | [] | [] |
| test3.csv | .csv | 108001 | 106466 | 1535 | [] | [] |
| test4.csv | .csv | 39601 | 38444 | 1157 | [] | [] |
| test5.csv | .csv | 92401 | 90224 | 2177 | [] | [] |

## Columns

### train1.csv

`time`, `P1_B2004`, `P1_B2016`, `P1_B3004`, `P1_B3005`, `P1_B4002`, `P1_B4005`, `P1_B400B`, `P1_B4022`, `P1_FCV01D`, `P1_FCV01Z`, `P1_FCV02D`, `P1_FCV02Z`, `P1_FCV03D`, `P1_FCV03Z`, `P1_FT01`, `P1_FT01Z`, `P1_FT02`, `P1_FT02Z`, `P1_FT03`, `P1_FT03Z`, `P1_LCV01D`, `P1_LCV01Z`, `P1_LIT01`, `P1_PCV01D`, `P1_PCV01Z`, `P1_PCV02D`, `P1_PCV02Z`, `P1_PIT01`, `P1_PIT02`, `P1_PP01AD`, `P1_PP01AR`, `P1_PP01BD`, `P1_PP01BR`, `P1_PP02D`, `P1_PP02R`, `P1_STSP`, `P1_TIT01`, `P1_TIT02`, `P2_24Vdc`, `P2_ASD`, `P2_AutoGO`, `P2_CO_rpm`, `P2_Emerg`, `P2_HILout`, `P2_MSD`, `P2_ManualGO`, `P2_OnOff`, `P2_RTR`, `P2_SIT01`, `P2_SIT02`, `P2_TripEx`, `P2_VT01`, `P2_VTR01`, `P2_VTR02`, `P2_VTR03`, `P2_VTR04`, `P2_VXT02`, `P2_VXT03`, `P2_VYT02`, `P2_VYT03`, `P3_FIT01`, `P3_LCP01D`, `P3_LCV01D`, `P3_LH`, `P3_LIT01`, `P3_LL`, `P3_PIT01`, `P4_HT_FD`, `P4_HT_LD`, `P4_HT_PO`, `P4_HT_PS`, `P4_LD`, `P4_ST_FD`, `P4_ST_GOV`, `P4_ST_LD`, `P4_ST_PO`, `P4_ST_PS`, `P4_ST_PT01`, `P4_ST_TT01`, `attack`, `attack_P1`, `attack_P2`, `attack_P3`

### train2.csv

`time`, `P1_B2004`, `P1_B2016`, `P1_B3004`, `P1_B3005`, `P1_B4002`, `P1_B4005`, `P1_B400B`, `P1_B4022`, `P1_FCV01D`, `P1_FCV01Z`, `P1_FCV02D`, `P1_FCV02Z`, `P1_FCV03D`, `P1_FCV03Z`, `P1_FT01`, `P1_FT01Z`, `P1_FT02`, `P1_FT02Z`, `P1_FT03`, `P1_FT03Z`, `P1_LCV01D`, `P1_LCV01Z`, `P1_LIT01`, `P1_PCV01D`, `P1_PCV01Z`, `P1_PCV02D`, `P1_PCV02Z`, `P1_PIT01`, `P1_PIT02`, `P1_PP01AD`, `P1_PP01AR`, `P1_PP01BD`, `P1_PP01BR`, `P1_PP02D`, `P1_PP02R`, `P1_STSP`, `P1_TIT01`, `P1_TIT02`, `P2_24Vdc`, `P2_ASD`, `P2_AutoGO`, `P2_CO_rpm`, `P2_Emerg`, `P2_HILout`, `P2_MSD`, `P2_ManualGO`, `P2_OnOff`, `P2_RTR`, `P2_SIT01`, `P2_SIT02`, `P2_TripEx`, `P2_VT01`, `P2_VTR01`, `P2_VTR02`, `P2_VTR03`, `P2_VTR04`, `P2_VXT02`, `P2_VXT03`, `P2_VYT02`, `P2_VYT03`, `P3_FIT01`, `P3_LCP01D`, `P3_LCV01D`, `P3_LH`, `P3_LIT01`, `P3_LL`, `P3_PIT01`, `P4_HT_FD`, `P4_HT_LD`, `P4_HT_PO`, `P4_HT_PS`, `P4_LD`, `P4_ST_FD`, `P4_ST_GOV`, `P4_ST_LD`, `P4_ST_PO`, `P4_ST_PS`, `P4_ST_PT01`, `P4_ST_TT01`, `attack`, `attack_P1`, `attack_P2`, `attack_P3`

### train3.csv

`time`, `P1_B2004`, `P1_B2016`, `P1_B3004`, `P1_B3005`, `P1_B4002`, `P1_B4005`, `P1_B400B`, `P1_B4022`, `P1_FCV01D`, `P1_FCV01Z`, `P1_FCV02D`, `P1_FCV02Z`, `P1_FCV03D`, `P1_FCV03Z`, `P1_FT01`, `P1_FT01Z`, `P1_FT02`, `P1_FT02Z`, `P1_FT03`, `P1_FT03Z`, `P1_LCV01D`, `P1_LCV01Z`, `P1_LIT01`, `P1_PCV01D`, `P1_PCV01Z`, `P1_PCV02D`, `P1_PCV02Z`, `P1_PIT01`, `P1_PIT02`, `P1_PP01AD`, `P1_PP01AR`, `P1_PP01BD`, `P1_PP01BR`, `P1_PP02D`, `P1_PP02R`, `P1_STSP`, `P1_TIT01`, `P1_TIT02`, `P2_24Vdc`, `P2_ASD`, `P2_AutoGO`, `P2_CO_rpm`, `P2_Emerg`, `P2_HILout`, `P2_MSD`, `P2_ManualGO`, `P2_OnOff`, `P2_RTR`, `P2_SIT01`, `P2_SIT02`, `P2_TripEx`, `P2_VT01`, `P2_VTR01`, `P2_VTR02`, `P2_VTR03`, `P2_VTR04`, `P2_VXT02`, `P2_VXT03`, `P2_VYT02`, `P2_VYT03`, `P3_FIT01`, `P3_LCP01D`, `P3_LCV01D`, `P3_LH`, `P3_LIT01`, `P3_LL`, `P3_PIT01`, `P4_HT_FD`, `P4_HT_LD`, `P4_HT_PO`, `P4_HT_PS`, `P4_LD`, `P4_ST_FD`, `P4_ST_GOV`, `P4_ST_LD`, `P4_ST_PO`, `P4_ST_PS`, `P4_ST_PT01`, `P4_ST_TT01`, `attack`, `attack_P1`, `attack_P2`, `attack_P3`

### test1.csv

`time`, `P1_B2004`, `P1_B2016`, `P1_B3004`, `P1_B3005`, `P1_B4002`, `P1_B4005`, `P1_B400B`, `P1_B4022`, `P1_FCV01D`, `P1_FCV01Z`, `P1_FCV02D`, `P1_FCV02Z`, `P1_FCV03D`, `P1_FCV03Z`, `P1_FT01`, `P1_FT01Z`, `P1_FT02`, `P1_FT02Z`, `P1_FT03`, `P1_FT03Z`, `P1_LCV01D`, `P1_LCV01Z`, `P1_LIT01`, `P1_PCV01D`, `P1_PCV01Z`, `P1_PCV02D`, `P1_PCV02Z`, `P1_PIT01`, `P1_PIT02`, `P1_PP01AD`, `P1_PP01AR`, `P1_PP01BD`, `P1_PP01BR`, `P1_PP02D`, `P1_PP02R`, `P1_STSP`, `P1_TIT01`, `P1_TIT02`, `P2_24Vdc`, `P2_ASD`, `P2_AutoGO`, `P2_CO_rpm`, `P2_Emerg`, `P2_HILout`, `P2_MSD`, `P2_ManualGO`, `P2_OnOff`, `P2_RTR`, `P2_SIT01`, `P2_SIT02`, `P2_TripEx`, `P2_VT01`, `P2_VTR01`, `P2_VTR02`, `P2_VTR03`, `P2_VTR04`, `P2_VXT02`, `P2_VXT03`, `P2_VYT02`, `P2_VYT03`, `P3_FIT01`, `P3_LCP01D`, `P3_LCV01D`, `P3_LH`, `P3_LIT01`, `P3_LL`, `P3_PIT01`, `P4_HT_FD`, `P4_HT_LD`, `P4_HT_PO`, `P4_HT_PS`, `P4_LD`, `P4_ST_FD`, `P4_ST_GOV`, `P4_ST_LD`, `P4_ST_PO`, `P4_ST_PS`, `P4_ST_PT01`, `P4_ST_TT01`, `attack`, `attack_P1`, `attack_P2`, `attack_P3`

### test2.csv

`time`, `P1_B2004`, `P1_B2016`, `P1_B3004`, `P1_B3005`, `P1_B4002`, `P1_B4005`, `P1_B400B`, `P1_B4022`, `P1_FCV01D`, `P1_FCV01Z`, `P1_FCV02D`, `P1_FCV02Z`, `P1_FCV03D`, `P1_FCV03Z`, `P1_FT01`, `P1_FT01Z`, `P1_FT02`, `P1_FT02Z`, `P1_FT03`, `P1_FT03Z`, `P1_LCV01D`, `P1_LCV01Z`, `P1_LIT01`, `P1_PCV01D`, `P1_PCV01Z`, `P1_PCV02D`, `P1_PCV02Z`, `P1_PIT01`, `P1_PIT02`, `P1_PP01AD`, `P1_PP01AR`, `P1_PP01BD`, `P1_PP01BR`, `P1_PP02D`, `P1_PP02R`, `P1_STSP`, `P1_TIT01`, `P1_TIT02`, `P2_24Vdc`, `P2_ASD`, `P2_AutoGO`, `P2_CO_rpm`, `P2_Emerg`, `P2_HILout`, `P2_MSD`, `P2_ManualGO`, `P2_OnOff`, `P2_RTR`, `P2_SIT01`, `P2_SIT02`, `P2_TripEx`, `P2_VT01`, `P2_VTR01`, `P2_VTR02`, `P2_VTR03`, `P2_VTR04`, `P2_VXT02`, `P2_VXT03`, `P2_VYT02`, `P2_VYT03`, `P3_FIT01`, `P3_LCP01D`, `P3_LCV01D`, `P3_LH`, `P3_LIT01`, `P3_LL`, `P3_PIT01`, `P4_HT_FD`, `P4_HT_LD`, `P4_HT_PO`, `P4_HT_PS`, `P4_LD`, `P4_ST_FD`, `P4_ST_GOV`, `P4_ST_LD`, `P4_ST_PO`, `P4_ST_PS`, `P4_ST_PT01`, `P4_ST_TT01`, `attack`, `attack_P1`, `attack_P2`, `attack_P3`

### test3.csv

`time`, `P1_B2004`, `P1_B2016`, `P1_B3004`, `P1_B3005`, `P1_B4002`, `P1_B4005`, `P1_B400B`, `P1_B4022`, `P1_FCV01D`, `P1_FCV01Z`, `P1_FCV02D`, `P1_FCV02Z`, `P1_FCV03D`, `P1_FCV03Z`, `P1_FT01`, `P1_FT01Z`, `P1_FT02`, `P1_FT02Z`, `P1_FT03`, `P1_FT03Z`, `P1_LCV01D`, `P1_LCV01Z`, `P1_LIT01`, `P1_PCV01D`, `P1_PCV01Z`, `P1_PCV02D`, `P1_PCV02Z`, `P1_PIT01`, `P1_PIT02`, `P1_PP01AD`, `P1_PP01AR`, `P1_PP01BD`, `P1_PP01BR`, `P1_PP02D`, `P1_PP02R`, `P1_STSP`, `P1_TIT01`, `P1_TIT02`, `P2_24Vdc`, `P2_ASD`, `P2_AutoGO`, `P2_CO_rpm`, `P2_Emerg`, `P2_HILout`, `P2_MSD`, `P2_ManualGO`, `P2_OnOff`, `P2_RTR`, `P2_SIT01`, `P2_SIT02`, `P2_TripEx`, `P2_VT01`, `P2_VTR01`, `P2_VTR02`, `P2_VTR03`, `P2_VTR04`, `P2_VXT02`, `P2_VXT03`, `P2_VYT02`, `P2_VYT03`, `P3_FIT01`, `P3_LCP01D`, `P3_LCV01D`, `P3_LH`, `P3_LIT01`, `P3_LL`, `P3_PIT01`, `P4_HT_FD`, `P4_HT_LD`, `P4_HT_PO`, `P4_HT_PS`, `P4_LD`, `P4_ST_FD`, `P4_ST_GOV`, `P4_ST_LD`, `P4_ST_PO`, `P4_ST_PS`, `P4_ST_PT01`, `P4_ST_TT01`, `attack`, `attack_P1`, `attack_P2`, `attack_P3`

### test4.csv

`time`, `P1_B2004`, `P1_B2016`, `P1_B3004`, `P1_B3005`, `P1_B4002`, `P1_B4005`, `P1_B400B`, `P1_B4022`, `P1_FCV01D`, `P1_FCV01Z`, `P1_FCV02D`, `P1_FCV02Z`, `P1_FCV03D`, `P1_FCV03Z`, `P1_FT01`, `P1_FT01Z`, `P1_FT02`, `P1_FT02Z`, `P1_FT03`, `P1_FT03Z`, `P1_LCV01D`, `P1_LCV01Z`, `P1_LIT01`, `P1_PCV01D`, `P1_PCV01Z`, `P1_PCV02D`, `P1_PCV02Z`, `P1_PIT01`, `P1_PIT02`, `P1_PP01AD`, `P1_PP01AR`, `P1_PP01BD`, `P1_PP01BR`, `P1_PP02D`, `P1_PP02R`, `P1_STSP`, `P1_TIT01`, `P1_TIT02`, `P2_24Vdc`, `P2_ASD`, `P2_AutoGO`, `P2_CO_rpm`, `P2_Emerg`, `P2_HILout`, `P2_MSD`, `P2_ManualGO`, `P2_OnOff`, `P2_RTR`, `P2_SIT01`, `P2_SIT02`, `P2_TripEx`, `P2_VT01`, `P2_VTR01`, `P2_VTR02`, `P2_VTR03`, `P2_VTR04`, `P2_VXT02`, `P2_VXT03`, `P2_VYT02`, `P2_VYT03`, `P3_FIT01`, `P3_LCP01D`, `P3_LCV01D`, `P3_LH`, `P3_LIT01`, `P3_LL`, `P3_PIT01`, `P4_HT_FD`, `P4_HT_LD`, `P4_HT_PO`, `P4_HT_PS`, `P4_LD`, `P4_ST_FD`, `P4_ST_GOV`, `P4_ST_LD`, `P4_ST_PO`, `P4_ST_PS`, `P4_ST_PT01`, `P4_ST_TT01`, `attack`, `attack_P1`, `attack_P2`, `attack_P3`

### test5.csv

`time`, `P1_B2004`, `P1_B2016`, `P1_B3004`, `P1_B3005`, `P1_B4002`, `P1_B4005`, `P1_B400B`, `P1_B4022`, `P1_FCV01D`, `P1_FCV01Z`, `P1_FCV02D`, `P1_FCV02Z`, `P1_FCV03D`, `P1_FCV03Z`, `P1_FT01`, `P1_FT01Z`, `P1_FT02`, `P1_FT02Z`, `P1_FT03`, `P1_FT03Z`, `P1_LCV01D`, `P1_LCV01Z`, `P1_LIT01`, `P1_PCV01D`, `P1_PCV01Z`, `P1_PCV02D`, `P1_PCV02Z`, `P1_PIT01`, `P1_PIT02`, `P1_PP01AD`, `P1_PP01AR`, `P1_PP01BD`, `P1_PP01BR`, `P1_PP02D`, `P1_PP02R`, `P1_STSP`, `P1_TIT01`, `P1_TIT02`, `P2_24Vdc`, `P2_ASD`, `P2_AutoGO`, `P2_CO_rpm`, `P2_Emerg`, `P2_HILout`, `P2_MSD`, `P2_ManualGO`, `P2_OnOff`, `P2_RTR`, `P2_SIT01`, `P2_SIT02`, `P2_TripEx`, `P2_VT01`, `P2_VTR01`, `P2_VTR02`, `P2_VTR03`, `P2_VTR04`, `P2_VXT02`, `P2_VXT03`, `P2_VYT02`, `P2_VYT03`, `P3_FIT01`, `P3_LCP01D`, `P3_LCV01D`, `P3_LH`, `P3_LIT01`, `P3_LL`, `P3_PIT01`, `P4_HT_FD`, `P4_HT_LD`, `P4_HT_PO`, `P4_HT_PS`, `P4_LD`, `P4_ST_FD`, `P4_ST_GOV`, `P4_ST_LD`, `P4_ST_PO`, `P4_ST_PS`, `P4_ST_PT01`, `P4_ST_TT01`, `attack`, `attack_P1`, `attack_P2`, `attack_P3`

