The Propane Knock Index Methane Number calculator API have the following requirements:
1. The available ships have been declared at the start of the code.(line 26)
2. The avauilable ships (ship_id) also has references in line 450 of the code.
3. If any ship identity has to be added or edited, it has be done in both above mentione lines viz., 26 & 450.
4. Similarly if any unique LNG tank configurations are on board apart from the ones that have declared already in the code, then new unique configurations have to be added in line 75.
5. The application works by first pushing daily data to CouchDB database.
6. SHIP_DB_MAP in line 80 contains the various names of databases where data for every ship is stored & populated on a daily basis.
7. The application will then fetch the data based on ship and date queried form relevant database pertaining to the queried vessel.
8. The application will then compute the values and return as json output for further analysis or analytics/ inferences.
9. Please note that the first row of the database for all ships is an Initialised data. It means that many of the values have to be declared. An example of the values that are to be decalred for the first row are shown. 
{
  "_id": "2025-04-01",
  "ship_id": "CMA CGM MONACO",
  "Date": "2025-04-01",
  "BDN": null,
  "ME_cons": 31.4905,
  "F.vap_cons": null,
  "GE_cons": 4.2824,
  "BLR_cons": null,
  "ROB_cal": 689,
  "ROB_cams": 677,
  "Discrepancy": 11,
  "Trim": -1,
  "List": 0,
  "Level_TK1": 3250,
  "Vap_temp_TK1": -121.3,
  "Liq_temp_TK1": -149.5,
  "Press_TK1": 1.19,
  "Level_TK2": 3100,
  "Vap_temp_TK2": -128.8,
  "Liq_temp_TK2": -150.2,
  "Press_TK2": 1.19,
  "CH4": null,
  "C2H6": null,
  "C3H8": null,
  "i-C4H10": null,
  "n-C4H10": null,
  "i-C5H12": null,
  "n-C5H12": null,
  "n-C6H14+": null,
  "N2": null,
  "CH4s": null,
  "C2H6s": null,
  "C3H8s": null,
  "i-C4H10s": null,
  "n-C4H10s": null,
  "i-C5H12s": null,
  "n-C5H12s": null,
  "n-C6H14+s": null,
  "N2s": null,
  "CH4c": 0.881303,
  "C2H6c": 0.092142,
  "C3H8c": 0.018065,
  "i-C4H10c": 0.00216,
  "n-C4H10c": 0.002512,
  "i-C5H12c": 0,
  "n-C5H12c": 0.000232,
  "n-C6H14+c": 0.000214,
  "N2c": 0,
  "CH4m": 606.9298,
  "C2H6m": 63.45559,
  "C3H8m": 12.44085,
  "i-C4H10m": 1.487275,
  "n-C4H10m": 1.729868,
  "i-C5H12m": 0,
  "n-C5H12m": 0.159892,
  "n-C6H14+m": 0.147239,
  "N2m": 0,
  "PKI": 7.27483,
  "MN": 70,
  "Min_MN": null,
  "Molar_Mass": 18.01216,
  "Density": 0.441808
}
10. If you are maintaining a month to month datasheet, then required values (shown above) of the last of the month are to be entered as intialised values for the 1st day of next month. 
11. It is essential for the 1st row to be initialised, the subsequent rows take reference from previous rows for calculations. Hence, the 1st row has some values declared statically and other columns for 1st row ara calculated based on it. 
12. In short, the calculation methodology for 1st row is different from the 2nd row onwards. These have been factroed in the codes.
13. for local deployment use http://localhost:8002/docs


