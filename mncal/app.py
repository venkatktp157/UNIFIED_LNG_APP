#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import re
import couchdb
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scipy.interpolate import griddata, RegularGridInterpolator, interp1d
from dotenv import load_dotenv
from config import DATA_DIR

# Load .env file
load_dotenv()

# Define Base Directory
# base_dir = os.path.dirname(os.path.abspath(__file__))
ship_dir = DATA_DIR

available_ships = {
    "MOUNT TOURMALINE": ["LNG_TK1", "LNG_TK2"],
    "MOUNT NOVATERRA" : ["LNG_TK1", "LNG_TK2"],    
    "MOUNT ANETO": ["LNG_TK1", "LNG_TK2"],
    "MOUNT TAI": ["LNG_TK1", "LNG_TK2"],
    "MOUNT OSSA": ["LNG_TK1", "LNG_TK2"],
    "MOUNT JADEITE": ["LNG_TK1", "LNG_TK2"],
    "MOUNT API": ["LNG_TK1", "LNG_TK2"],
    "MOUNT AMELIOR": ["LNG_TK1", "LNG_TK2"],
    "MOUNT HENG": ["LNG_TK1", "LNG_TK2"],
    "MOUNT GOWER": ["LNG_TK1", "LNG_TK2"],
    "MOUNT GAEA": ["LNG_TK1", "LNG_TK2"],
    "MOUNT COOK": ["LNG_TK1", "LNG_TK2"],
    "MOUNT ARARAT": ["LNG_TK1", "LNG_TK2"],    
    "CMA CGM ARCTIC" : ["LNG_TK"],
    "CMA CGM BALI" : ["LNG_TK"],
    "CMA CGM DIGNITY" : ["LNG_TK"],
    "CMA CGM HOPE" : ["LNG_TK"],
    "CMA CGM IGUACU" : ["LNG_TK"],
    "CMA CGM INTEGRITY" : ["LNG_TK"],
    "CMA CGM LIBERTY" : ["LNG_TK"],
    "CMA CGM TENERE" : ["LNG_TK"],
    "CMA CGM PRIDE" : ["LNG_TK"],
    "CMA CGM SCANDOLA" : ["LNG_TK"],
    "CMA CGM SYMI" : ["LNG_TK"],
    "CMA CGM UNITY" : ["LNG_TK"],
    "ZIM ARIES" : ["LNG_TANK"],
    "ZIM GEMINI" : ["LNG_TANK"],
    "ZIM SCORPIO" : ["LNG_TANK"],
    "QUETZAL" : ["LNGAS_TK"],
    "COPAN" : ["LNGAS_TK"],
    "CMA CGM DAYTONA": ["LNG_TK1", "LNG_TK2"],
    "CMA CGM INDIANAPOLIS": ["LNG_TK1", "LNG_TK2"],
    "CMA CGM MONACO": ["LNG_TK1", "LNG_TK2"],
    "CMA CGM SILVERSTONE": ["LNG_TK1", "LNG_TK2"],
    "CMA CGM MONZA": ["LNG_TK1", "LNG_TK2"],
    "LAKE HERMAN": ["LNG_TK1", "LNG_TK2"],
    "LAKE ANNECY": ["LNG_TK1", "LNG_TK2"],
    "LAKE LUGU": ["LNG_TK1", "LNG_TK2"],
    "LAKE QARAOUN": ["LNG_TK1", "LNG_TK2"],
    "LAKE SAINT ANNE": ["LNG_TK1", "LNG_TK2"],
    "LAKE TRAVIS": ["LNG_TK1", "LNG_TK2"],
    "LAKE TAZAWA": ["LNG_TK1", "LNG_TK2"],
    "ATLANTIC JADE": ["LNG_TK1", "LNG_TK2"],
    "ATLANTIC EMERALD": ["LNG_TK1", "LNG_TK2"],
    "STARWAY": ["LNG_TK1", "LNG_TK2"],
    "GREENWAY": ["LNG_TK1", "LNG_TK2"],
}

# Flag to identify LNG tanks (no corrections needed)
lng_tanks = ["LNG_TK"]
lng_tks = ["LNG_TANK"]
LNG_TK  = ["LNGAS_TK"]

SHIP_DB_MAP = {
    "CMA CGM MONACO": "monaco",      # custom mapping for legacy reasons
    "GREENWAY": "greenway",
    "CMA CGM PRIDE": "pride",
    # Add more mappings...
}
def get_db_name_for_ship(ship_id):
    return SHIP_DB_MAP.get(ship_id, ship_id)

app = FastAPI(
    title="CONSOLIDATED PKI MN Calculator API (FastAPI only)",
    description="REST API for PKI/MN/LNG calculations - CouchDB backend, FastAPI only",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory where ship data is stored    
def get_tank_data_path(ship_id, tank_id):
    ship_data_dir = os.path.join(ship_dir, ship_id)
    if tank_id in lng_tanks:
        return {
            "list_table": os.path.join(ship_data_dir, f"list_table_{tank_id}.csv"),
            "trim_table": os.path.join(ship_data_dir, f"trim_table_{tank_id}.csv")
        } 

    elif tank_id in LNG_TK:
        return {
            "list_table": os.path.join(ship_data_dir, f"list_table_{tank_id}.csv"),
            "trim_table": os.path.join(ship_data_dir, f"trim_table_{tank_id}.csv")
        } 

    elif tank_id in lng_tks:
        return {
            "list_table": os.path.join(ship_data_dir, f"list_table_{tank_id}.csv"),
            "trim_table": os.path.join(ship_data_dir, f"trim_table_{tank_id}.csv"),
            "volume_table": os.path.join(ship_data_dir, f"volume_table_{tank_id}.csv")
        }   
    else:
        return {
            "volume_table": os.path.join(ship_data_dir, f"volume_table_{tank_id}.csv"),
            "list_table": os.path.join(ship_data_dir, f"list_table_{tank_id}.csv"),
            "trim_table": os.path.join(ship_data_dir, f"trim_table_{tank_id}.csv"),
            "temp_table": os.path.join(ship_data_dir, f"temp_table_{tank_id}.csv"),
            "press_table": os.path.join(ship_data_dir, f"press_table_{tank_id}.csv")            
        }

# Function to load min-max values from datasets
def get_range_values(ship_id, tank_id):
    tank_paths = get_tank_data_path(ship_id, tank_id)      
    
    if tank_id in lng_tanks:
        required_files = ["list_table", "trim_table"]
        for file_key in required_files:
            if file_key not in tank_paths or not os.path.exists(tank_paths[file_key]):
                raise HTTPException(status_code=400, detail=f"Missing data file: {file_key} for ship {ship_id} and tank {tank_id}.")
        
            
        # Load datasets
        level_list_df = pd.read_csv(tank_paths["list_table"])
        level_trim_df = pd.read_csv(tank_paths["trim_table"])

        # Function to extract numerical values from column names
        def ext_val(columns, prefix):
            values = []
            for col in columns:
                match = re.search(rf"{prefix}([-+]?\d*\.?\d+)", col)
                if match:
                    values.append(float(match.group(1)))
            return values

        # Extract values
        list_values = ext_val(level_list_df.columns[1:], "list_")
        trim_values = ext_val(level_trim_df.columns[1:], "trim_")

        # Get min-max values
        level_min, level_max = float(level_trim_df["level"].min()), float(level_trim_df["level"].max())
        list_min, list_max = (min(list_values), max(list_values)) if list_values else (None, None)
        trim_min, trim_max = (min(trim_values), max(trim_values)) if trim_values else (None, None)
        temp_min, temp_max = -163.0, 20.0
        press_min, press_max = -1.0, 6.0

        return level_min, level_max, list_min, list_max, trim_min, trim_max, temp_min, temp_max, press_min, press_max

    elif tank_id in LNG_TK:
        required_files = ["list_table", "trim_table"]
        for file_key in required_files:
            if file_key not in tank_paths or not os.path.exists(tank_paths[file_key]):
                raise HTTPException(status_code=400, detail=f"Missing data file: {file_key} for ship {ship_id} and tank {tank_id}.")

        # Load datasets
        level_list_df = pd.read_csv(tank_paths["list_table"])
        level_trim_df = pd.read_csv(tank_paths["trim_table"])

        # Function to extract numerical values from column names
        def ext_val(columns, prefix):
            values = []
            for col in columns:
                match = re.search(rf"{prefix}([-+]?\d*\.?\d+)", col)
                if match:
                    values.append(float(match.group(1)))
            return values

        # Extract values
        list_values = ext_val(level_list_df.columns[1:], "list_")
        trim_values = ext_val(level_trim_df.columns[1:], "trim_")

        # Get min-max values
        level_min, level_max = float(level_trim_df["level"].min()), float(level_trim_df["level"].max())
        list_min, list_max = (min(list_values), max(list_values)) if list_values else (None, None)
        trim_min, trim_max = (min(trim_values), max(trim_values)) if trim_values else (None, None)
        temp_min, temp_max = -165.0, 20.0
        press_min, press_max = 0, 0.7

        return level_min, level_max, list_min, list_max, trim_min, trim_max, temp_min, temp_max, press_min, press_max


    elif tank_id in lng_tks:
        required_files = ["list_table", "trim_table", "volume_table"]
        for file_key in required_files:
            if file_key not in tank_paths or not os.path.exists(tank_paths[file_key]):
               raise HTTPException(status_code=400, detail=f"Missing data file: {file_key} for ship {ship_id} and tank {tank_id}.")

        # Load datasets
        level_list_df = pd.read_csv(tank_paths["list_table"])
        level_trim_df = pd.read_csv(tank_paths["trim_table"])
        level_volume_df = pd.read_csv(tank_paths["volume_table"])

        # Function to extract numerical values from column names
        def ext_vals(columns, prefix):
            values = []
            for col in columns:
                match = re.search(rf"{prefix}([-+]?\d*\.?\d+)", col)
                if match:
                    values.append(float(match.group(1)))
            return values

        # Extract values
        list_values = ext_vals(level_list_df.columns[1:], "list_")
        trim_values = ext_vals(level_trim_df.columns[1:], "trim_")        

        # Get min-max values
        level_min, level_max = float(level_volume_df["level"].min()), float(level_volume_df["level"].max())
        list_min, list_max = (min(list_values), max(list_values)) if list_values else (None, None)
        trim_min, trim_max = (min(trim_values), max(trim_values)) if trim_values else (None, None)
        temp_min, temp_max = -163.0, 20.0
        press_min, press_max = -1.0, 6.0

        return level_min, level_max, list_min, list_max, trim_min, trim_max, temp_min, temp_max, press_min, press_max      

    else:
        # Ensure all required files exist
        required_files = ["volume_table", "list_table", "trim_table", "temp_table", "press_table"]
        for file_key in required_files:
            if file_key in tank_paths and not os.path.exists(tank_paths[file_key]):
                raise HTTPException(status_code=400, detail=f"Missing data file: {file_key} for ship {ship_id} and tank {tank_id}.")

        # Load datasets
        level_volume_df = pd.read_csv(tank_paths["volume_table"])
        level_list_df = pd.read_csv(tank_paths["list_table"])
        level_trim_df = pd.read_csv(tank_paths["trim_table"])
        level_temp_df = pd.read_csv(tank_paths["temp_table"])
        level_press_df = pd.read_csv(tank_paths["press_table"])       

        # Function to extract numerical values from column names
        def extract_values(columns, prefix):
            values = []
            for col in columns:
                match = re.search(rf"{prefix}([-+]?\d*\.?\d+)", col)
                if match:
                    values.append(float(match.group(1)))                    
            return values

        # Extract values
        list_values = extract_values(level_list_df.columns[1:], "list_")
        trim_values = extract_values(level_trim_df.columns[1:], "trim_")
        temp_values = extract_values(level_temp_df.columns[1:], "temp_")
        press_values = extract_values(level_press_df.columns[1:], "press_")       

        # Get min-max values
        level_min, level_max = float(level_volume_df["level"].min()), float(level_volume_df["level"].max())
        list_min, list_max = (min(list_values), max(list_values)) if list_values else (None, None)
        trim_min, trim_max = (min(trim_values), max(trim_values)) if trim_values else (None, None)
        temp_min, temp_max = (min(temp_values), max(temp_values)) if temp_values else (None, None)
        press_min, press_max = (min(press_values), max(press_values)) if press_values else (None, None)

        return level_min, level_max, list_min, list_max, trim_min, trim_max, temp_min, temp_max, press_min, press_max

# Function to compute corrected values
def compute_corrected_values(ship_id, tank_id, level, list_, trim_, temp_, press_):
    tank_paths = get_tank_data_path(ship_id, tank_id)

    if tank_paths is None:
        return None, None

    if tank_id in lng_tanks:
        required_files = ["list_table", "trim_table"] 
        for file_key in required_files:
            if not os.path.exists(tank_paths[file_key]):
                raise HTTPException(status_code=400, detail=f"Missing data file: {file_key}")

        # Load datasets
        level_list_df = pd.read_csv(tank_paths["list_table"])
        level_trim_df = pd.read_csv(tank_paths["trim_table"])

         # Extract values
        level_values_1 = level_list_df["level"].values
        level_values_2 = level_trim_df["level"].values
        list_values = np.array([float(col.replace("list_", "")) for col in level_list_df.columns[1:]])
        trim_values = np.array([float(col.replace("trim_", "")) for col in level_trim_df.columns[1:]])

         # Create interpolators
        level_list_interpolator = RegularGridInterpolator(
            (level_values_1, list_values), level_list_df.iloc[:, 1:].values, method="linear"
        )
        level_trim_interpolator = RegularGridInterpolator(
            (level_values_2, trim_values), level_trim_df.iloc[:, 1:].values, method="linear"
        ) 

        # Interpolate values
        list_correction = level_list_interpolator([[level, list_]])[0]
        corrected_level = level + list_correction
        corrected_volume = level_trim_interpolator([[corrected_level, trim_]])[0]    

        return round(corrected_level, 2), round(corrected_volume, 2)   

    elif tank_id in LNG_TK:
        required_files = ["list_table", "trim_table"] 
        for file_key in required_files:
            if not os.path.exists(tank_paths[file_key]):
                raise HTTPException(status_code=400, detail=f"Missing data file: {file_key}")

        # Load datasets
        level_list_df = pd.read_csv(tank_paths["list_table"])
        level_trim_df = pd.read_csv(tank_paths["trim_table"])

         # Extract values
        level_values_1 = level_list_df["level"].values
        level_values_2 = level_trim_df["level"].values
        # volume_values = level_volume_df["volume"].values
        list_values = np.array([float(col.replace("list_", "")) for col in level_list_df.columns[1:]])
        trim_values = np.array([float(col.replace("trim_", "")) for col in level_trim_df.columns[1:]])

         # Create interpolators
        level_list_interpolator = RegularGridInterpolator(
            (level_values_1, list_values), level_list_df.iloc[:, 1:].values, method="linear"
        )
        level_trim_interpolator = RegularGridInterpolator(
            (level_values_2, trim_values), level_trim_df.iloc[:, 1:].values, method="linear"
        ) 

        # Interpolate values
        list_correction = level_list_interpolator([[level, list_]])[0]
        #trim_correction = level_trim_interpolator([[level, trim_]])[0]       

        corrected_level = level + list_correction

        corrected_volume = level_trim_interpolator([[corrected_level, trim_]])[0]    

        return round(corrected_level, 2), round(corrected_volume, 2)  

    elif tank_id in lng_tks:
        required_files = ["list_table", "trim_table", "volume_table"] 
        for file_key in required_files:
            if not os.path.exists(tank_paths[file_key]):
                raise HTTPException(status_code=400, detail=f"Missing data file: {file_key}")

        # Load datasets
        level_list_df = pd.read_csv(tank_paths["list_table"])
        level_trim_df = pd.read_csv(tank_paths["trim_table"])
        level_volume_df = pd.read_csv(tank_paths["volume_table"])

         # Extract values
        level_values = level_volume_df["level"].values
        volume_values = level_volume_df["volume"].values
        list_values = np.array([float(col.replace("list_", "")) for col in level_list_df.columns[1:]])
        trim_values = np.array([float(col.replace("trim_", "")) for col in level_trim_df.columns[1:]])

         # Create interpolators
        level_list_interpolator = RegularGridInterpolator(
            (level_values, list_values), level_list_df.iloc[:, 1:].values, method="linear"
        )
        level_trim_interpolator = RegularGridInterpolator(
            (level_values, trim_values), level_trim_df.iloc[:, 1:].values, method="linear"
        ) 

        level_volume_interpolator = RegularGridInterpolator(
            (level_values,), volume_values, method="linear"
        ) 

        # Interpolate values
        list_correction = level_list_interpolator([[level, list_]])[0]
        trim_correction = level_trim_interpolator([[level, trim_]])[0]       
        corrected_level = level + list_correction + trim_correction
        corrected_volume = level_volume_interpolator([[corrected_level]])[0]    

        return round(corrected_level, 2), round(corrected_volume, 2)           

    else:
        required_files = ["volume_table", "list_table", "trim_table", "temp_table", "press_table"] 
        for file_key in required_files:
            if not os.path.exists(tank_paths[file_key]):
                raise HTTPException(status_code=400, detail=f"Missing data file: {file_key}")


        # Load datasets
        level_volume_df = pd.read_csv(tank_paths["volume_table"])
        level_list_df = pd.read_csv(tank_paths["list_table"])
        level_trim_df = pd.read_csv(tank_paths["trim_table"])
        level_temp_df = pd.read_csv(tank_paths["temp_table"])
        level_press_df = pd.read_csv(tank_paths["press_table"])       

        # Extract values
        level_values = level_volume_df["level"].values
        volume_values = level_volume_df["volume"].values
        list_values = np.array([float(col.replace("list_", "")) for col in level_list_df.columns[1:]])
        trim_values = np.array([float(col.replace("trim_", "")) for col in level_trim_df.columns[1:]])
        temp_values = np.array([float(col.replace("temp_", "")) for col in level_temp_df.columns[1:]])
        press_values = np.array([float(col.replace("press_", "")) for col in level_press_df.columns[1:]])    

        # Create interpolators
        level_list_interpolator = RegularGridInterpolator(
            (level_values, list_values), level_list_df.iloc[:, 1:].values, method="linear"
        )
        level_trim_interpolator = RegularGridInterpolator(
            (level_values, trim_values), level_trim_df.iloc[:, 1:].values, method="linear"
        )
        level_temp_interpolator = RegularGridInterpolator(
            (level_values, temp_values), level_temp_df.iloc[:, 1:].values, method="linear"
        )
        level_press_interpolator = RegularGridInterpolator(
            (level_values, press_values), level_press_df.iloc[:, 1:].values, method="linear"
        )

        level_volume_interpolator = RegularGridInterpolator(
            (level_values,), volume_values, method="linear"
        ) 

        # Interpolate values
        list_correction = level_list_interpolator([[level, list_]])[0]
        trim_correction = level_trim_interpolator([[level, trim_]])[0]
        temp_correction = level_temp_interpolator([[level, temp_]])[0]
        press_correction = level_press_interpolator([[level, press_]])[0]    
        corrected_level = level + list_correction + trim_correction + temp_correction + press_correction    
        corrected_volume = level_volume_interpolator([[corrected_level]])[0]    

        return round(corrected_level, 2), round(corrected_volume, 2)   

# Helper function for interpolation
def interpolate_value(df, x_col, y_col, x_val):
    if x_val <= df[x_col].min():
        return df[y_col].iloc[0]
    elif x_val >= df[x_col].max():
        return df[y_col].iloc[-1]
    else:
        return np.interp(x_val, df[x_col], df[y_col])  


def calculate_columnwise(df, ship_id, tank_ids):    
    # PKI MN CALCULATIONS
    # Use ship_id here safely    
    tank_ids = available_ships[ship_id]        
        
    # Set ship-specific parameters
    if ship_id in ["MOUNT TOURMALINE", "MOUNT NOVATERRA"]:   #209k_bulk ships
        BOG_max = 500
        LNG_TK1_cap = 3175.139
        LNG_TK2_cap = 3180.121
        identity = "209k_bulk"
    elif ship_id in ["MOUNT ANETO", "MOUNT TAI", "MOUNT OSSA", "MOUNT JADEITE", "MOUNT API", "MOUNT AMELIOR", "MOUNT HENG", 
                    "MOUNT GOWER", "MOUNT GAEA", "MOUNT COOK", "MOUNT ARARAT"]: #210k_bulk
        BOG_max = 500
        LNG_TK1_cap = 3181.546
        LNG_TK2_cap = 3179.732
        identity = "210k_bulk"       
    elif ship_id in ["CMA CGM ARCTIC", "CMA CGM BALI", "CMA CGM DIGNITY", "CMA CGM HOPE", "CMA CGM IGUACU",
                    "CMA CGM INTEGRITY", "CMA CGM LIBERTY", "CMA CGM PRIDE", "CMA CGM TENERE", "CMA CGM SCANDOLA",
                    "CMA CGM SYMI", "CMA CGM UNITY"]:   #CMA_cont 
        BOG_max = 500
        LNG_TK1_cap = 12448.3
        identity = "CMA_cont"
    elif ship_id in ["ZIM ARIES", "ZIM GEMINI", "ZIM SCORPIO"]: #ZIM_cont
        BOG_max = 1200
        LNG_TK1_cap = 6125.285    
        identity = "ZIM_cont"       
    elif ship_id in ["CMA CGM DAYTONA", "CMA CGM INDIANAPOLIS", "CMA CGM MONACO", "CMA CGM SILVERSTONE",
                    "CMA CGM MONZA", "LAKE HERMAN", "LAKE ANNECY", "LAKE LUGU", "LAKE QARAOUN", "LAKE SAINT ANNE",
                    "LAKE TRAVIS", "LAKE TAZAWA"]: #PCTC
        BOG_max = 600
        LNG_TK1_cap = 2013.699
        LNG_TK2_cap = 2014.748
        identity = "PCTC"
    elif ship_id in ["ATLANTIC JADE", "ATLANTIC EMERALD"]:   #110K_tanker
        BOG_max = 1200
        LNG_TK1_cap = 2324.113
        LNG_TK2_cap = 2322.097
        identity = "110k_tanker"
    elif ship_id in ["STARWAY", "GREENWAY"]:   #150K_tanker
        BOG_max = 1200
        LNG_TK1_cap = 2570.133
        LNG_TK2_cap = 2571.517
        identity = "150k_tanker"              
    elif ship_id in ["QUETZAL", "COPAN"]:   #1400TEU_cont
        BOG_max = 500
        LNG_TK1_cap = 1613
        identity = "1400TEU_cont"     


    # ========== CONFIGURATION ==============
    COUCHDB_USER = os.getenv("COUCHDB_USER")
    COUCHDB_PASS = os.getenv("COUCHDB_PASS")
    COUCHDB_HOST = os.getenv("COUCHDB_HOST", "localhost")
    COUCHDB_PORT = os.getenv("COUCHDB_PORT", "5984")

    COUCHDB_URL = f"http://{COUCHDB_USER}:{COUCHDB_PASS}@{COUCHDB_HOST}:{COUCHDB_PORT}/"
    db_name = get_db_name_for_ship(ship_id)

    couch = couchdb.Server(COUCHDB_URL)
    if db_name in couch:
        db = couch[db_name]
    else:
        raise HTTPException(status_code=404, detail=f"CouchDB database '{db_name}' does not exist.")

    # List of all column names from your dataset
    columns = [
        "ship_id","Date", "BDN", "ME_cons", "F.vap_cons", "GE_cons", "BLR_cons", "ROB_cal", "ROB_cams", 
        "Discrepancy", "Trim", "List", "Level_TK1", "Vap_temp_TK1", "Liq_temp_TK1", "Press_TK1", 
        "Level_TK2", "Vap_temp_TK2", "Liq_temp_TK2", "Press_TK2", "CH4", "C2H6", "C3H8", 
        "i-C4H10", "n-C4H10", "i-C5H12", "n-C5H12", "n-C6H14+", "N2", "CH4s", "C2H6s", 
        "C3H8s", "i-C4H10s", "n-C4H10s", "i-C5H12s", "n-C5H12s", "n-C6H14+s", "N2s", "CH4c", 
        "C2H6c", "C3H8c", "i-C4H10c", "n-C4H10c", "i-C5H12c", "n-C5H12c", "n-C6H14+c", "N2c", 
        "CH4m", "C2H6m", "C3H8m", "i-C4H10m", "n-C4H10m", "i-C5H12m", "n-C5H12m", "n-C6H14+m", 
        "N2m", "PKI", "MN", "Min_MN", "Molar_Mass", "Density", "MM_sum", "C6+", "CH4f", 
        "CH4^2", "CH4^3", "CH4^4", "C2H6f", "C2H6^2", "C2H6^3", "C2H6^4", "C3H8f", "C3H8^2", 
        "C3H8^3", "C3H8^4", "N-C4f", "N-C4^2", "N-C4^3", "N-C4^4", "I-C4f", "I-C4^2", 
        "I-C4^3", "I-C4^4", "N-C5f", "N-C5^2", "N-C5^3", "N-C5^4", "I-C5f", "I-C5^2", 
        "I-C5^3", "I-C5^4", "NEC5f", "NEC5^2", "NEC5^3", "NEC5^4", "N2f", "N2^2", "N2^3", 
        "N2^4", "CO2f", "CO2^2", "CO2^3", "CO2^4", "COf", "CO^2", "CO^3", "CO^4", "H2f", 
        "H2^2", "H2^3", "H2^4", "CH4*C2H6", "CH4*C3H8", "CH4*N-C4", "(CH4*N-C4)^2", 
        "CH4*I-C4", "(CH4*I-C4)^2", "CH4*N-C5", "CH4*I-C5", "CH4*NEC5", "CH4*N2", 
        "CH4*CO2", "(CH4*CO2)^2", "CH4*CO", "CH4*H2", "CH4*(H2^2)", "(CH4^2)*H2", 
        "C2H6*C3H8", "C2H6*N-C4", "C2H6*I-C4", "C2H6*N-C5", "C2H6*I-C5", "C2H6*NEC5", 
        "C2H6*N2", "(C2H6^2)*N2", "C2H6*(N2^2)", "C2H6*CO2", "C2H6*CO", "C2H6*H2", 
        "C3H8*N-C4", "C3H8*I-C4", "C3H8*N-C5", "C3H8*(N-C5^2)", "(C3H8^2)*N-C5)", 
        "C3H8*I-C5", "C3H8*NEC5", "C3H8*N2", "C3H8*CO2", "C3H8*CO", "(C3H8^2)*CO", 
        "C3H8*H2", "N-C4*I-C4", "N-C4*N-C5", "N-C4*(N-C5^2)", "(N-C4^2)*N-C5)", 
        "N-C4*I-C5", "N-C4*NEC5", "N-C4*N2", "N-C4*CO2", "N-C4*CO", "N-C4*H2", 
        "I-C4*N-C5", "I-C4*I-C5", "I-C4*NEC5", "I-C4*N2", "I-C4*CO2", "I-C4*CO", 
        "I-C4*H2", "N-C5*I-C5", "N-C5*NEC5", "N-C5*N2", "N-C5*CO2", "(N-C5^2)*CO2", 
        "N-C5*CO", "(N-C5^2)*CO", "N-C5*H2", "(N-C5^2)*H2", "I-C5*NEC5", "I-C5*N2", 
        "I-C5*CO2", "(I-C5^2)*CO2", "I-C5*CO", "I-C5*H2", "NEC5*N2", "NEC5*CO2", 
        "(NEC5^2)*CO2", "NEC5*CO", "NEC5*H2", "N2*CO2", "(N2^2)*CO2", "N2*CO", 
        "(N2^2)*CO", "N2*(CO^2)", "N2*H2", "CO2*CO", "(CO2*CO)^2", "CO2*H2", 
        "(CO2*H2)^2", "CO*H2", "(CO*H2)^2", "(N-C5*H2)^2", "CO*(H2^2)", "k1_TK1", 
        "k1_TK2", "k2_TK1", "k2_TK2", "Methane_temp1", "Ethane_temp1", "Propane_temp1", 
        "Iso-butan_temp1", "Butane_temp1", "Iso-penta_temp1", "Pentane_temp1", 
        "Nitrogen_temp1", "Methane_temp2", "Ethane_temp2", "Propane_temp2", 
        "Iso-butan_temp2", "Butane_temp2", "Iso-penta_temp2", "Pentane_temp2", 
        "Nitrogen_temp2", "tank1_level", "tank1_liquid_vol", "tank2_level", 
        "tank2_liquid_vol", "total_liq_vol", "k1_corr", "k2_corr", "Methane_corr", 
        "Ethane_corr", "Propane_corr", "Iso-butan_corr", "Butane_corr", 
        "Iso-penta_corr", "Pentane_corr", "Nitrogen_corr", "comp_sum", 
        "tank1_vap_vol", "tank1_total_vol", "tank2_vap_vol", "tank2_total_vol", 
        "total_volume"
    ]
    
    def fetch_data():
        docs = [db[doc_id] for doc_id in db]
        df = pd.DataFrame(docs)
        return df

    # Initialize empty DataFrame with all columns
    df = pd.DataFrame(0, index=[0], columns=columns)

    try:
        couch_df = fetch_data().fillna(0)
        if couch_df.empty:
            raise HTTPException(status_code=404, detail="No matching documents found for given ship_id and tank_ids")

        # Ensure all expected columns are present
        for col in columns:
            if col not in couch_df.columns:
                couch_df[col] = 0

        # Reorder columns
        df = couch_df[columns]        

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching or aligning data: {str(e)}")   
      
    # Validate required columns
    required_cols = ['Date', 'List', 'Trim']
    # Add tank-specific columns based on available tanks
    for i, tank_id in enumerate(tank_ids, start=1):
        required_cols.extend([
            f'Level_TK{i}',
            f'Vap_temp_TK{i}',
            f'Liq_temp_TK{i}',
            f'Press_TK{i}'
        ])

    # Get range values for each tank
    tank_ranges = {}
    for tank_num, tank_id in enumerate(tank_ids, start=1):
        ranges = get_range_values(ship_id, tank_id)
        if ranges:
            tank_ranges[tank_id] = {
                'level_min': ranges[0],
                'level_max': ranges[1],
                'list_min': ranges[2],
                'list_max': ranges[3],
                'trim_min': ranges[4],
                'trim_max': ranges[5],
                'temp_min': ranges[6],
                'temp_max': ranges[7],
                'press_min': ranges[8],
                'press_max': ranges[9]
            } 
    
    # Temperature & Pressure Corrections
    # base_dir = os.path.dirname(os.path.abspath(__file__))
    ship_dir = DATA_DIR
    
    ship_data_dir = os.path.join(ship_dir, ship_id)
    tank_paths = {
            "tempcorr_table": os.path.join(ship_data_dir, f"tempcorr_table_{tank_ids[0]}.csv"),
            "presscorr_table": os.path.join(ship_data_dir, f"presscorr_table_{tank_ids[0]}.csv")
        }

    # Load CSV files
    tempcorr_df = pd.read_csv(tank_paths["tempcorr_table"])
    presscorr_df = pd.read_csv(tank_paths["presscorr_table"]) 

    #ROB CALCULATIONS---------------------------------------------------------------------------------------------

    # Ensure all required columns exist and are numeric
    required_cols = ['BDN', 'ME_cons', 'F.vap_cons', 'GE_cons', 'BLR_cons']
    for col in required_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0.0  # Create column if missing   

    #----------------------------------------------------------------------------------------------------------------
    # Convert percentage columns to fractions
    component_cols = ['CH4', 'C2H6', 'C3H8', 'i-C4H10', 'n-C4H10', 'i-C5H12', 'n-C5H12', 'n-C6H14+', 'N2']
    for col in component_cols:
        df[f'{col}s'] = df[col] / 100

    for i in range(1, len(df)):
        df.at[i, 'ROB_cal'] = (df.at[i-1, 'ROB_cal'] + 
                            df.at[i, 'BDN'] - 
                            df.at[i, 'ME_cons'] - 
                            df.at[i, 'F.vap_cons'] - 
                            df.at[i, 'GE_cons'] - 
                            df.at[i, 'BLR_cons'])

    # Calculate for subsequent rows
    for i in range(1, len(df)):
        # Calculate total consumption
        total_cons = df.at[i, 'ME_cons'] + df.at[i, 'F.vap_cons'] + df.at[i, 'GE_cons'] + df.at[i, 'BLR_cons']        

        # Calculate each component's molar mass
        for comp in ['CH4', 'C2H6', 'C3H8', 'i-C4H10', 'n-C4H10', 'i-C5H12', 'n-C5H12', 'n-C6H14+']:
            prev_m = df.at[i-1, f'{comp}m']
            prev_c = df.at[i-1, f'{comp}c']
            new_supply = df.at[i, 'BDN'] * df.at[i, f'{comp}s']

            # Different handling for CH4
            if comp == 'CH4':
                consumed = (df.at[i, 'ME_cons'] + df.at[i, 'F.vap_cons']) * prev_c + df.at[i, 'GE_cons'] + df.at[i, 'BLR_cons']
            else:
                consumed = (df.at[i, 'ME_cons'] + df.at[i, 'F.vap_cons']) * prev_c

            df.at[i, f'{comp}m'] = prev_m - consumed + new_supply

        # Special handling for N2
        prev_N2m = df.at[i-1, 'N2m']
        N2_consumed = (df.at[i, 'GE_cons'] + df.at[i, 'BLR_cons']) * (prev_N2m / df.at[i-1, 'ROB_cal'])
        N2_supplied = df.at[i, 'BDN'] * df.at[i, 'N2s']
        df.at[i, 'N2m'] = max(prev_N2m - N2_consumed + N2_supplied, 0)

        # Calculate current compositions
        current_rob = df.at[i, 'ROB_cal']
        for comp in ['CH4', 'C2H6', 'C3H8', 'i-C4H10', 'n-C4H10', 'i-C5H12', 'n-C5H12', 'n-C6H14+', 'N2']:
            if current_rob > 0:
                df.at[i, f'{comp}c'] = df.at[i, f'{comp}m'] / current_rob
            else:
                df.at[i, f'{comp}c'] = 0

    # Calculate MM_sum and C6+
    df['MM_sum'] = df[['CH4m', 'C2H6m', 'C3H8m', 'i-C4H10m', 'n-C4H10m', 'i-C5H12m', 'n-C5H12m', 'n-C6H14+m', 'N2m']].sum(axis=1)
    df['C6+'] = df['n-C6H14+c'] * 0.3       


    # Final rounding
    round_cols = ['ROB_cal', 'CH4c', 'CH4m', 'C2H6c', 'C2H6m', 'C3H8c', 'C3H8m', 
                'i-C4H10c', 'i-C4H10m', 'n-C4H10c', 'n-C4H10m', 'i-C5H12c', 'i-C5H12m',
                'n-C5H12c', 'n-C5H12m', 'n-C6H14+c', 'n-C6H14+m', 'N2c', 'N2m', 'MM_sum', 'C6+']

    for col in round_cols:
        if col in ['CH4c', 'C2H6c', 'C3H8c', 'i-C4H10c', 'n-C4H10c', 'i-C5H12c', 'n-C5H12c', 'n-C6H14+c', 'N2c']:
            df[col] = df[col].round(6)
        elif col in ['C6+']:
            df[col] = df[col].round(7)
        else:
            df[col] = df[col].round(3)
    #---------------------------------------------------------------------------------------------------------

    #Calculate Specific formula cells
    #populate various constants for composition

    coefficients = [569.285536016002, -650.8543394907, 64.3595752573862, 17.2149592220536, -645.099966662855, 
                    694.229376857102, -675.381075231165, 1474.79079137333, 499.39849265152,-576.665945472394, 
                    252.19367406028, 593.958975466507, 934.46627322324, -86.8723570770238, -20418.9067673979, 
                    633286.561358521, 735.223884113728, -3182.61439337967, 20945.1867250219, 159067.868032595, 
                    2571.93079360535, 10516.4941092275, -770539.377197693, 28633475.5865654, -3582.96784435379, 
                    0, 403155.950864334, -11917333.8379329, 1123.39636709865, 1679.7280752481, -172182.649067176, 
                    3467918.60746699, -469.428097827742, 352.688107288763, -220.491687402358, 1419.68005396242, 
                    -953.460328339263, 1148.48725868228, -601.339855375907, 448.125565457084, -5813.75996390021, 
                    5511.72102582867, 1647.04306584326, -3471.24152555425, -2012.525219063, 2059.63157031447, 
                    -313.277150726788, 957.327608016344, 201.788909592169, -865.856657223225, -1210.2275419324, 
                    1331.55552369645, -1023.2781474703, 1550.09518461258, -2811.67740432523, 3363.98150506356, 
                    -1534.52567488723, -1.05397332930609, 473.57476410971, -308.25901022921, 5356.4335705495, 
                    1227.10772949701, 253.206759621511, 326.009795302013, 0, -437.695363730406, -109.983789902769, 
                    -1870.34746500563, 3909.50906076245, -886.578525827322, 968.887620927515, 267.47276619196, 
                    337.464863958288, 1431.95011699315, 6463.14444295627, 2974.72929658495, -118.490180710956, 0, 
                    -1734.80568239427, 127551.642193201, 11318.4183950722, 3318.96820819338, 0, 13.3453378124692, 
                    292.275289330565, 5403.50260794829, 2333.82346342921, 2067.29242460978, 3500.70282852274, 
                    -4737.32849494999, 525591.310711326, 297556.039242685, 6095.05998875087, -953.002183779388, 0, 
                    -103.571484346062, 5869.19050652774, 2377.69485624119, 5056.60309163761, 6619.27877637044, 
                    -1363.96101644841, 14.8038957999724, 211.752602673394, 5786.32525717488, 2567.6536314925, 
                    12268.283772748, 0, -1573.68893770625, -898.466856535774, -42401.4111391824, 3985.11042051103, 
                    48265.3191033737, 0, 99313.9508434517, 3773.44926785397, 4490.67830032675, 5122.00993545509, 
                    -28087.8481864326, 10248.3408254232, 6575.39711806826, -642.170828416611, 0, -11320.1126899481, 
                    4772.67730118682, 1108.92638475254, 1156.20032716021, 359.342203118816, 6076.81809291631, 
                    389.853153629781, 367.319351280689, 2616.21956431342, 6557.3763494187, 1824.58587937403, 
                    3034.7413460668, -1664.28094074521, 8006.50820723109, 884.14262538453, 0, 0]

    #calculate individual compositions of each constituents- equivalent to number of coefficients displayed above

    df['CH4f'] = df['CH4c'] - df['C6+']
    df['CH4^2'] = df['CH4f']**2
    df['CH4^3'] = df['CH4f']**3
    df['CH4^4'] = df['CH4f']**4

    df['C2H6f'] = df['C2H6c']
    df['C2H6^2'] = df['C2H6f']**2
    df['C2H6^3'] = df['C2H6f']**3
    df['C2H6^4'] = df['C2H6f']**4

    df['C3H8f'] = df['C3H8c']
    df['C3H8^2'] = df['C3H8f']**2
    df['C3H8^3'] = df['C3H8f']**3
    df['C3H8^4'] = df['C3H8f']**4

    df['N-C4f'] = df['n-C4H10c']
    df['N-C4^2'] = df['N-C4f']**2
    df['N-C4^3'] = df['N-C4f']**3
    df['N-C4^4'] = df['N-C4f']**4   

    df['I-C4f'] = df['i-C4H10c']
    df['I-C4^2'] = df['I-C4f']**2
    df['I-C4^3'] = df['I-C4f']**3
    df['I-C4^4'] = df['I-C4f']**4

    df['N-C5f'] = df['n-C5H12c'] + 1.3 * df['n-C6H14+c']
    df['N-C5^2'] = df['N-C5f']**2
    df['N-C5^3'] = df['N-C5f']**3
    df['N-C5^4'] = df['N-C5f']**4   

    df['I-C5f'] = df['i-C5H12c']
    df['I-C5^2'] = df['I-C5f']**2
    df['I-C5^3'] = df['I-C5f']**3
    df['I-C5^4'] = df['I-C5f']**4   

    df['NEC5f'] = 0
    df['NEC5^2'] = df['NEC5f']**2
    df['NEC5^3'] = df['NEC5f']**3
    df['NEC5^4'] = df['NEC5f']**4

    df['N2f'] = df['N2c']
    df['N2^2'] = df['N2f']**2
    df['N2^3'] = df['N2f']**3
    df['N2^4'] = df['N2f']**4

    df['CO2f'] = 0
    df['CO2^2'] = df['CO2f']**2
    df['CO2^3'] = df['CO2f']**3
    df['CO2^4'] = df['CO2f']**4

    df['COf'] = 0
    df['CO^2'] = df['COf']**2
    df['CO^3'] = df['COf']**3
    df['CO^4'] = df['COf']**4

    df['H2f'] = 0
    df['H2^2'] = df['H2f']**2
    df['H2^3'] = df['H2f']**3
    df['H2^4'] = df['H2f']**4

    df['CH4*C2H6'] = df['CH4f'] * df['C2H6f']
    df['CH4*C3H8'] = df['CH4f'] * df['C3H8f']

    df['CH4*N-C4'] = df['CH4f'] * df['N-C4f']
    df['(CH4*N-C4)^2']  = df['CH4*N-C4']**2

    df['CH4*I-C4'] = df['CH4f'] * df['I-C4f']
    df['(CH4*I-C4)^2'] =df['CH4*I-C4']**2

    df['CH4*N-C5'] = df['CH4f'] * df['N-C5f']
    df['CH4*I-C5'] = df['CH4f'] * df['I-C5f']

    df['CH4*NEC5'] = 0
    df['CH4*N2'] = df['CH4f'] * df['N2f']

    df['CH4*CO2'] = 0
    df['(CH4*CO2)^2'] = df['CH4*CO2']**2

    df['CH4*CO'] = 0
    df['CH4*H2'] = 0
    df['CH4*(H2^2)'] = df['CH4f'] * (df['H2f']**2)
    df['(CH4^2)*H2'] = (df['CH4f']**2) * df['H2f']

    df['C2H6*C3H8'] = df['C2H6f'] * df['C3H8f']
    df['C2H6*N-C4'] = df['C2H6f'] * df['N-C4f']
    df['C2H6*I-C4'] = df['C2H6f'] * df['I-C4f']
    df['C2H6*N-C5'] = df['C2H6f'] * df['N-C5f']    
    df['C2H6*I-C5'] = df['C2H6f'] * df['I-C5f']

    df['C2H6*NEC5'] = 0
    df['C2H6*N2'] = df['C2H6f'] * df['N2f']
    df['(C2H6^2)*N2'] = (df['C2H6f']**2) * df['N2f']
    df['C2H6*(N2^2)'] = df['C2H6f'] * (df['N2f']**2)

    df['C2H6*CO2'] = df['C2H6f'] * df['CO2f']
    df['C2H6*CO'] = df['C2H6f'] * df['COf']
    df['C2H6*H2'] = df['C2H6f'] * df['H2f']

    df['C3H8*N-C4'] = df['C3H8f'] * df['N-C4f']
    df['C3H8*I-C4'] = df['C3H8f'] * df['I-C4f']
    df['C3H8*N-C5'] = df['C3H8f'] * df['N-C5f']    
    df['C3H8*(N-C5^2)'] = df['C3H8f'] * (df['N-C5f']**2)
    df['(C3H8^2)*N-C5)'] = (df['C3H8f']**2) * df['N-C5f']**2

    df['C3H8*I-C5'] = df['C3H8f'] * df['I-C5f']
    df['C3H8*NEC5'] = df['C3H8f'] * df['NEC5f']
    df['C3H8*N2'] = df['C3H8f'] * df['N2f']

    df['C3H8*CO2'] = df['C3H8f'] * df['CO2f']
    df['C3H8*CO'] = df['C3H8f'] * df['COf']
    df['(C3H8^2)*CO'] = (df['C3H8f']**2) * df['COf']
    df['C3H8*H2'] = df['C3H8f'] * df['H2f']

    df['N-C4*I-C4'] = df['N-C4f'] * df['I-C4f']
    df['N-C4*N-C5'] = df['N-C4f'] * df['N-C5f']
    df['N-C4*(N-C5^2)'] = df['N-C4f'] * (df['N-C5f']**2)
    df['(N-C4^2)*N-C5)'] = (df['N-C4f']**2) * df['N-C5f']

    df['N-C4*I-C5'] = df['N-C4f'] * df['I-C5f']
    df['N-C4*NEC5'] = df['N-C4f'] * df['NEC5f']
    df['N-C4*N2'] = df['N-C4f'] * df['N2f']

    df['N-C4*CO2'] = df['N-C4f'] * df['CO2f']
    df['N-C4*CO'] = df['N-C4f'] * df['COf']
    df['N-C4*H2'] = df['N-C4f'] * df['H2f']

    df['I-C4*N-C5'] = df['I-C4f'] * df['N-C5f']
    df['I-C4*I-C5'] = df['I-C4f'] * df['I-C5f']
    df['I-C4*NEC5'] = df['I-C4f'] * df['NEC5f']
    df['I-C4*N2'] = df['I-C4f'] * df['N2f']

    df['I-C4*CO2'] = df['I-C4f'] * df['CO2f']
    df['I-C4*CO'] = df['I-C4f'] * df['COf']
    df['I-C4*H2'] = df['I-C4f'] * df['H2f']

    df['N-C5*I-C5'] = df['N-C5f'] * df['I-C5f']
    df['N-C5*NEC5'] = df['N-C5f'] * df['NEC5f']
    df['N-C5*N2'] = df['N-C5f'] * df['N2f']

    df['N-C5*CO2'] = df['N-C5f'] * df['CO2f']
    df['(N-C5^2)*CO2'] = (df['N-C5f']**2) * df['CO2f']

    df['N-C5*CO'] = df['N-C5f'] * df['COf']
    df['(N-C5^2)*CO'] = (df['N-C5f']**2) * df['COf']

    df['N-C5*H2'] = df['N-C5f'] * df['H2f']
    df['(N-C5^2)*H2'] = (df['N-C5f']**2) * df['H2f']

    df['I-C5*NEC5'] = df['I-C5f'] * df['NEC5f']
    df['I-C5*N2'] = df['I-C5f'] * df['N2f']

    df['I-C5*CO2'] = df['I-C5f'] * df['CO2f']
    df['(I-C5^2)*CO2'] = (df['I-C5f']**2) * df['CO2f']

    df['I-C5*CO'] = df['I-C5f'] * df['COf']
    df['I-C5*H2'] = df['I-C5f'] * df['H2f']
    df['NEC5*N2'] = df['NEC5f'] * df['N2f']

    df['NEC5*CO2'] = df['NEC5f'] * df['CO2f']
    df['(NEC5^2)*CO2'] = (df['NEC5f']**2) * df['CO2f']

    df['NEC5*CO'] = df['NEC5f'] * df['COf']
    df['NEC5*H2'] = df['NEC5f'] * df['H2f']

    df['N2*CO2'] = df['N2f'] * df['CO2f']
    df['(N2^2)*CO2'] = (df['N2f']**2) * df['CO2f']

    df['N2*CO'] = df['N2f'] * df['COf']
    df['(N2^2)*CO'] = (df['N2f']**2) * df['COf']
    df['N2*(CO^2)'] = df['N2f'] * (df['COf']**2)
    df['N2*H2'] = df['N2f'] * df['H2f']

    df['CO2*CO'] = df['CO2f'] * df['COf']
    df['(CO2*CO)^2'] = (df['CO2f'] * df['COf'])**2

    df['CO2*H2'] = df['CO2f'] * df['H2f']
    df['(CO2*H2)^2'] = (df['CO2f'] * df['H2f'])**2

    df['CO*H2'] = df['COf'] * df['H2f']
    df['(CO*H2)^2'] = (df['COf'] * df['H2f'])**2

    df['(N-C5*H2)^2'] = (df['N-C5f'] * df['H2f'])**2
    df['CO*(H2^2)'] = df['COf'] * (df['H2f']**2)


    target_columns = ['CH4f','CH4^2','CH4^3','CH4^4','C2H6f','C2H6^2','C2H6^3','C2H6^4','C3H8f','C3H8^2','C3H8^3','C3H8^4',
                        'N-C4f','N-C4^2','N-C4^3','N-C4^4','I-C4f','I-C4^2','I-C4^3','I-C4^4','N-C5f','N-C5^2','N-C5^3','N-C5^4',
                        'I-C5f','I-C5^2','I-C5^3','I-C5^4','NEC5f','NEC5^2','NEC5^3','NEC5^4','N2f','N2^2','N2^3','N2^4','CO2f',
                        'CO2^2','CO2^3','CO2^4','COf','CO^2','CO^3','CO^4','H2f','H2^2','H2^3','H2^4','CH4*C2H6','CH4*C3H8',
                        'CH4*N-C4','(CH4*N-C4)^2','CH4*I-C4','(CH4*I-C4)^2','CH4*N-C5','CH4*I-C5','CH4*NEC5','CH4*N2','CH4*CO2',
                        '(CH4*CO2)^2','CH4*CO','CH4*H2','CH4*(H2^2)','(CH4^2)*H2','C2H6*C3H8','C2H6*N-C4','C2H6*I-C4','C2H6*N-C5',
                        'C2H6*I-C5','C2H6*NEC5','C2H6*N2','(C2H6^2)*N2','C2H6*(N2^2)','C2H6*CO2','C2H6*CO','C2H6*H2','C3H8*N-C4',
                        'C3H8*I-C4','C3H8*N-C5','C3H8*(N-C5^2)','(C3H8^2)*N-C5)','C3H8*I-C5','C3H8*NEC5','C3H8*N2','C3H8*CO2',
                        'C3H8*CO','(C3H8^2)*CO','C3H8*H2','N-C4*I-C4','N-C4*N-C5','N-C4*(N-C5^2)','(N-C4^2)*N-C5)','N-C4*I-C5',
                        'N-C4*NEC5','N-C4*N2','N-C4*CO2','N-C4*CO','N-C4*H2','I-C4*N-C5','I-C4*I-C5','I-C4*NEC5','I-C4*N2',
                        'I-C4*CO2','I-C4*CO','I-C4*H2','N-C5*I-C5','N-C5*NEC5','N-C5*N2','N-C5*CO2','(N-C5^2)*CO2','N-C5*CO',
                        '(N-C5^2)*CO','N-C5*H2','(N-C5^2)*H2','I-C5*NEC5','I-C5*N2','I-C5*CO2','(I-C5^2)*CO2','I-C5*CO',
                        'I-C5*H2','NEC5*N2','NEC5*CO2','(NEC5^2)*CO2','NEC5*CO','NEC5*H2','N2*CO2','(N2^2)*CO2','N2*CO',
                        '(N2^2)*CO','N2*(CO^2)','N2*H2','CO2*CO','(CO2*CO)^2','CO2*H2','(CO2*H2)^2','CO*H2','(CO*H2)^2',
                        '(N-C5*H2)^2','CO*(H2^2)']

    # Calculate PKI- Select only the target columns and multiply
    df.loc[1:,'PKI'] = (df[target_columns] * coefficients).sum(axis=1)

    #Calculate Methane Number:

    a1 = -9.757977
    a2 = 1.484961
    a3 = -0.139533
    a4 = 0.007031306
    a5 = -0.000177003
    a6 = 1.75121E-06
    b = 100

    df.loc[1:,'MN'] = (b + (df['PKI']*a1) 
                + (df['PKI']**2)*a2 
                + (df['PKI']**3)*a3 
                + (df['PKI']**4)*a4 
                + (df['PKI']**5)*a5
                + (df['PKI']**6)*a6)

    df.loc[1:,'MN'] = df.loc[1:,'MN'].round(0)    

    df['Min_MN'] = 75.0

    # Calculate the Molar Mass
    df.loc[1:,'Molar_Mass'] = (df['CH4c'] * 16.04246 + 
                        df['C2H6c'] * 30.06904 + 
                        df['C3H8c'] * 44.09562 + 
                        df['i-C4H10c'] * 58.12220 + 
                        df['n-C4H10c'] * 58.12220 + 
                        df['i-C5H12c'] * 72.14878 + 
                        df['n-C5H12c'] * 72.14878 + 
                        df['n-C6H14+c'] * 86.17536 + 
                        df['N2c'] * 28.01340)

    # K1 & K2 Calculations.
    #Extracting k1 values:------------------------------------------------------------------------------------

    # 1. Create the reference DataFrame from your table
    data = {
        'Molar_Mass': [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        'temp_180': [-0.01, 0.13, 0.25, 0.37, 0.47, 0.55, 0.64, 0.72, 0.81, 0.88, 0.95, 1.01, 1.06, 1.11, 1.16],
        'temp_175': [-0.01, 0.15, 0.29, 0.41, 0.52, 0.62, 0.72, 0.82, 0.92, 1.00, 1.07, 1.13, 1.18, 1.23, 1.29],
        'temp_170': [-0.01, 0.16, 0.33, 0.45, 0.59, 0.70, 0.81, 0.92, 1.04, 1.12, 1.19, 1.26, 1.32, 1.37, 1.43],
        'temp_165': [-0.01, 0.18, 0.37, 0.51, 0.67, 0.79, 0.90, 1.02, 1.16, 1.25, 1.33, 1.41, 1.47, 1.54, 1.60],
        'temp_160': [-0.01, 0.21, 0.41, 0.58, 0.76, 0.89, 1.01, 1.15, 1.30, 1.41, 1.50, 1.58, 1.64, 1.72, 1.79],
        'temp_155': [-0.01, 0.24, 0.47, 0.67, 0.86, 1.00, 1.17, 1.33, 1.47, 1.58, 1.68, 1.78, 1.84, 1.92, 2.00],
        'temp_150': [-0.01, 0.28, 0.56, 0.76, 0.98, 1.13, 1.32, 1.53, 1.66, 1.78, 1.89, 1.99, 2.06, 2.15, 2.24],
        'temp_145': [-0.01, 0.33, 0.66, 0.87, 1.10, 1.29, 1.52, 1.68, 1.87, 2.00, 2.13, 2.24, 2.32, 2.42, 2.51],
        'temp_140': [-0.01, 0.38, 0.76, 1.01, 1.30, 1.45, 1.71, 1.84, 2.13, 2.27, 2.41, 2.53, 2.62, 2.73, 2.83]
    }

    ref_df = pd.DataFrame(data)

    # 2. Reshape to long format for interpolation
    temp_cols = [col for col in ref_df.columns if col.startswith('temp_')]
    ref_long = ref_df.melt(id_vars=['Molar_Mass'], 
                            value_vars=temp_cols,
                            var_name='temp_str', 
                            value_name='k1')

    # Convert temperature strings to numerical values
    ref_long['temp'] = ref_long['temp_str'].str.extract(r'temp_(\d+)').astype(float) * -1

    # 3. Interpolation function
    def interpolate_k1(molar_mass, temp):
        import numpy as np 
        # 1. Return 0 if input is NaN or 0
        if pd.isna(molar_mass) or pd.isna(temp) or molar_mass == 0 or temp == 0:
            return 0

        # 2. Clip values to reference range (16 ≤ Molar_Mass ≤ 30, -180 ≤ Temp ≤ -140)
        molar_mass = np.clip(molar_mass, 16, 30)
        temp = np.clip(temp, -180, -140)        

        points = ref_long[['Molar_Mass', 'temp']].values
        values = ref_long['k1'].values
        return griddata(points, values, (molar_mass, temp), method='linear')  

        # 4. Fallback to 0 if interpolation fails
        return result if not np.isnan(result) else 0    

    # 4. Apply interpolation to your input data
    if len(tank_ids) > 1:
        df['k1_TK1'] = df.apply(lambda row: interpolate_k1(row['Molar_Mass'], row['Liq_temp_TK1']), axis=1)
        df['k1_TK2'] = df.apply(lambda row: interpolate_k1(row['Molar_Mass'], row['Liq_temp_TK2']), axis=1)
    else:
        df['k1_TK1'] = df.apply(lambda row: interpolate_k1(row['Molar_Mass'], row['Liq_temp_TK1']), axis=1)           

    #Extracting k2 values:-----------------------------------------------------------------------------------

    # 1. Create the reference DataFrame from your table
    data1 = {
        'Molar_Mass': [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        'temp_180': [0.00, 0.11, 0.26, 0.40, 0.56, 0.67, 0.78, 0.88, 0.98, 1.07, 1.15, 1.22, 1.31, 1.38, 1.47],
        'temp_175': [-0.01, 0.15, 0.32, 0.47, 0.62, 0.76, 0.90, 1.03, 1.13, 1.22, 1.31, 1.40, 1.50, 1.59, 1.68],
        'temp_170': [-0.01, 0.21, 0.39, 0.57, 0.71, 0.87, 1.01, 1.15, 1.27, 1.38, 1.50, 1.61, 1.72, 1.83, 1.93],
        'temp_165': [-0.01, 0.29, 0.53, 0.71, 0.86, 1.01, 1.16, 1.30, 1.45, 1.61, 1.74, 1.87, 1.99, 2.12, 2.24],
        'temp_160': [-0.02, 0.46, 0.67, 0.88, 1.06, 1.16, 1.27, 1.42, 1.60, 1.89, 2.04, 2.19, 2.33, 2.48, 2.63],
        'temp_155': [-0.03, 0.68, 0.84, 1.13, 1.33, 1.48, 1.65, 1.85, 2.06, 2.28, 2.44, 2.60, 2.77, 2.95, 3.12],
        'temp_150': [-0.04, 0.91, 1.05, 1.39, 1.62, 1.85, 2.09, 2.33, 2.58, 2.73, 2.92, 3.10, 3.31, 3.51, 3.72],
        'temp_145': [-0.05, 1.21, 1.34, 1.76, 2.03, 2.26, 2.51, 2.81, 3.11, 3.29, 3.48, 3.71, 3.95, 4.19, 4.45],
        'temp_140': [-0.07, 1.60, 1.80, 2.22, 2.45, 2.79, 3.13, 3.49, 3.74, 3.97, 4.19, 4.46, 4.74, 5.03, 5.34]
    }

    ref_df1 = pd.DataFrame(data1)

    # 2. Reshape to long format for interpolation
    temp_cols1 = [col for col in ref_df1.columns if col.startswith('temp_')]
    ref_long1 = ref_df1.melt(id_vars=['Molar_Mass'], 
                            value_vars=temp_cols1,
                            var_name='temp_str', 
                            value_name='k2')

    # Convert temperature strings to numerical values
    ref_long1['temp'] = ref_long1['temp_str'].str.extract(r'temp_(\d+)').astype(float) * -1    

    # 3. Interpolation function
    def interpolate_k2(molar_mass, temp):
        import numpy as np 
        # 1. Return 0 if input is NaN or 0
        if pd.isna(molar_mass) or pd.isna(temp) or molar_mass == 0 or temp == 0:
            return 0

        # 2. Clip values to reference range (16 ≤ Molar_Mass ≤ 30, -180 ≤ Temp ≤ -140)
        molar_mass = np.clip(molar_mass, 16, 30)
        temp = np.clip(temp, -180, -140)        

        points = ref_long1[['Molar_Mass', 'temp']].values
        values = ref_long1['k2'].values
        return griddata(points, values, (molar_mass, temp), method='linear')  

        # 4. Fallback to 0 if interpolation fails
        return result if not np.isnan(result) else 0

    # 4. Apply interpolation to your input data
    if len(tank_ids) > 1:
        df['k2_TK1'] = df.apply(lambda row: interpolate_k2(row['Molar_Mass'], row['Liq_temp_TK1']), axis=1)
        df['k2_TK2'] = df.apply(lambda row: interpolate_k2(row['Molar_Mass'], row['Liq_temp_TK2']), axis=1)

    else:
        df['k2_TK1'] = df.apply(lambda row: interpolate_k2(row['Molar_Mass'], row['Liq_temp_TK1']), axis=1)

    #Density calculations----------------------------------------------------------------------------------
    # 1. Load your component reference data (same as before)
    component_data = {
        'Temperature': [-180, -175, -170, -165, -160, -155, -150, -145, -140],
        'Methane': [0.0360, 0.0360, 0.0370, 0.0380, 0.0380, 0.0390, 0.0400, 0.0400, 0.0410],
        'Ethane': [0.0463, 0.0467, 0.0471, 0.0475, 0.0479, 0.0484, 0.0488, 0.0493, 0.0497],
        'Propane': [0.0607, 0.0612, 0.0616, 0.0620, 0.0625, 0.0630, 0.0634, 0.0639, 0.0644],
        'Iso-butan': [0.0764, 0.0769, 0.0774, 0.0779, 0.0784, 0.0789, 0.0794, 0.0799, 0.0804],
        'Butane': [0.0750, 0.0755, 0.0759, 0.0764, 0.0769, 0.0774, 0.0778, 0.0783, 0.0788],
        'Iso-penta': [0.0896, 0.0901, 0.0906, 0.0912, 0.0917, 0.0923, 0.0928, 0.0934, 0.0939],
        'Pentane': [0.0895, 0.0900, 0.0905, 0.0911, 0.0916, 0.0921, 0.0926, 0.0932, 0.0937],
        'Nitrogen': [0.0384, 0.0399, 0.0418, 0.0440, 0.0470, 0.0510, 0.0559, 0.0618, 0.0691]
    }
    component_df = pd.DataFrame(component_data)

    # 2. Create interpolation functions for each component
    interpolators = {}
    components = component_df.columns[1:]  # All columns except Temperature

    for component in components:
        interpolators[component] = interp1d(
            component_df['Temperature'],
            component_df[component],
            kind='linear',
            bounds_error=False,
            fill_value=0
        )

    # 4. Function to interpolate components for a temperature series
    def interpolate_components(temperature_series, suffix=''):
        """Interpolate all components for a temperature series and return as DataFrame"""
        result = pd.DataFrame()
        for component in components:
            result[f'{component}{suffix}'] = interpolators[component](temperature_series)
        return result

    # 5. Interpolate for both temperature columns
    if len(tank_ids) > 1:
        components_temp1 = interpolate_components(df['Liq_temp_TK1'], suffix='_temp1')
        components_temp2 = interpolate_components(df['Liq_temp_TK2'], suffix='_temp2')

    else:
        components_temp1 = interpolate_components(df['Liq_temp_TK1'], suffix='_temp1')

    # 6. Combine with original dataframe
    if len(tank_ids) > 1:
        df = pd.concat([df, components_temp1, components_temp2], axis=1)

    else:
        df = pd.concat([df, components_temp1], axis=1)
    #--------------------------------------------------------------------------------------------------------------------

    # LIQUID VOLUME PROCESSING
    liquid_results = []
    for i, row in df.iterrows():
        tank_calcs = {}
        for tank_num, tank_id in enumerate(tank_ids, start=1):
            # Define all column names needed for liquid calculation
            level_col = f'Level_TK{tank_num}'
            liq_temp_col = f'Liq_temp_TK{tank_num}'
            press_col = f'Press_TK{tank_num}'
            vap_temp_col = f'Vap_temp_TK{tank_num}'  # Added for corrected values calculation

            # Skip if no level data
            if pd.isna(row[level_col]):
                continue

            # Get tank capacity
            tank_capacity = LNG_TK1_cap if (len(tank_ids) == 1 or tank_num == 1) else LNG_TK2_cap

            # Get corrected level and volume
            ranges = tank_ranges.get(tank_id, {})
            level = max(min(row[level_col], ranges.get('level_max', row[level_col])), 
                        ranges.get('level_min', row[level_col]))
            list_ = max(min(row['List'], ranges.get('list_max', row['List'])), 
                        ranges.get('list_min', row['List']))
            trim_ = max(min(row['Trim'], ranges.get('trim_max', row['Trim'])), 
                        ranges.get('trim_min', row['Trim']))

            # Now using properly defined vap_temp_col
            corrected_level, corrected_volume = compute_corrected_values(
                ship_id, tank_id, level, list_, trim_, 
                row[vap_temp_col], row[press_col]  # These columns now exist
            )

            if corrected_level is None:
                continue

            # Apply corrections
            temp_correction = interpolate_value(tempcorr_df, 'Temp', 'tcorr', row[liq_temp_col])
            press_correction = interpolate_value(presscorr_df, 'Press', 'pcorr', row[press_col])
            liquid_volume = corrected_volume * temp_correction * press_correction

            tank_calcs.update({
                f'tank{tank_num}_level': corrected_level,
                f'tank{tank_num}_liquid_vol': liquid_volume,
                f'_tank{tank_num}_capacity': tank_capacity  # Store capacity per tank
            })

        liquid_results.append({**row.to_dict(), **tank_calcs})

    df1 = pd.DataFrame(liquid_results)

    if len(tank_ids) > 1:
        df1['total_liq_vol'] = df1['tank1_liquid_vol'] + df1['tank2_liquid_vol']

        df1['k1_corr'] = (df1['k1_TK1'] * df1['tank1_liquid_vol'] + df1['k1_TK2'] * df1['tank2_liquid_vol'])/df1['total_liq_vol']*0.001

        df1['k2_corr'] = (df1['k2_TK1'] * df1['tank1_liquid_vol'] + df1['k2_TK2'] * df1['tank2_liquid_vol'])/df1['total_liq_vol']*0.001

        df1['Methane_corr'] = ((df1['Methane_temp1'] * df1['tank1_liquid_vol'] 
                                + df1['Methane_temp2'] * df1['tank2_liquid_vol'])/df1['total_liq_vol'])*df1['CH4c']

        df1['Ethane_corr'] = ((df1['Ethane_temp1'] * df1['tank1_liquid_vol'] 
                                + df1['Ethane_temp2'] * df1['tank2_liquid_vol'])/df1['total_liq_vol'])*df1['C2H6c']

        df1['Propane_corr'] = ((df1['Propane_temp1'] * df1['tank1_liquid_vol'] 
                                + df1['Propane_temp2'] * df1['tank2_liquid_vol'])/df1['total_liq_vol'])*df1['C3H8c']

        df1['Iso-butan_corr'] = ((df1['Iso-butan_temp1'] * df1['tank1_liquid_vol'] 
                                + df1['Iso-butan_temp2'] * df1['tank2_liquid_vol'])/df1['total_liq_vol'])*df1['i-C4H10c']

        df1['Butane_corr'] = ((df1['Butane_temp1'] * df1['tank1_liquid_vol'] 
                                + df1['Butane_temp2'] * df1['tank2_liquid_vol'])/df1['total_liq_vol'])*df1['n-C4H10c']

        df1['Iso-penta_corr'] = ((df1['Iso-penta_temp1'] * df1['tank1_liquid_vol'] 
                                + df1['Iso-penta_temp2'] * df1['tank2_liquid_vol'])/df1['total_liq_vol'])*df1['i-C5H12c']

        df1['Pentane_corr'] = ((df1['Pentane_temp1'] * df1['tank1_liquid_vol'] 
                                + df1['Pentane_temp2'] * df1['tank2_liquid_vol'])/df1['total_liq_vol'])*df1['n-C5H12c']

        df1['Nitrogen_corr'] = ((df1['Nitrogen_temp1'] * df1['tank1_liquid_vol'] 
                                + df1['Nitrogen_temp2'] * df1['tank2_liquid_vol'])/df1['total_liq_vol'])*df1['N2c']

        df1['comp_sum'] = (df1['Methane_corr'] + df1['Ethane_corr'] + df1['Propane_corr']
                            + df1['Iso-butan_corr'] + df1['Butane_corr'] + df1['Iso-penta_corr']
                            + df1['Pentane_corr'] + df1['Nitrogen_corr'])

    else:
        df1['total_liq_vol'] = df1['tank1_liquid_vol'] 

        df1['k1_corr'] = df1['k1_TK1'] * 0.001

        df1['k2_corr'] = df1['k2_TK1'] * 0.001

        df1['Methane_corr'] = df1['Methane_temp1'] * df1['CH4c']

        df1['Ethane_corr'] = df1['Ethane_temp1'] * df1['C2H6c']

        df1['Propane_corr'] = df1['Propane_temp1'] * df1['C3H8c']

        df1['Iso-butan_corr'] = df1['Iso-butan_temp1'] * df1['i-C4H10c']

        df1['Butane_corr'] = df1['Butane_temp1'] * df1['n-C4H10c']

        df1['Iso-penta_corr'] = df1['Iso-penta_temp1'] * df1['i-C5H12c']

        df1['Pentane_corr'] = df1['Pentane_temp1'] * df1['n-C5H12c']

        df1['Nitrogen_corr'] = df1['Nitrogen_temp1'] * df1['N2c']

        df1.loc[1:,'comp_sum'] = (df1['Methane_corr'] + df1['Ethane_corr'] + df1['Propane_corr']
                            + df1['Iso-butan_corr'] + df1['Butane_corr'] + df1['Iso-penta_corr']
                            + df1['Pentane_corr'] + df1['Nitrogen_corr'])


    # DENSITY CALCULATIONS:-----------------------------------------------------------------------------------------
    # Calculate intermediate terms (for all rows, but we'll only use from row 1)
    a = ((df1['k2_corr'] - df1['k1_corr']) * df1['N2c'] / 0.0425)
    b = (df1['k1_corr'] + a * df1['CH4c'])
    c = (df1['comp_sum'] - b)  # Compute for all rows (we'll mask later)

    # Initialize density column (if not already present)
    if 'Density' not in df1:
        df1['Density'] = np.nan  # or default value

    # SAFEST APPROACH - Convert to numpy arrays for calculation (from row 1 onwards)
    molar_mass_values = df1.loc[1:, 'Molar_Mass'].values
    c_values = c.loc[1:].values  # Only compute from row 1

    # Initialize output array with zeros (for rows 1+)
    density_values = np.zeros_like(molar_mass_values)

    # Only perform division where c is not zero (for rows 1+)
    mask = c_values != 0
    density_values[mask] = molar_mass_values[mask] / c_values[mask]

    # Handle infinite values and negative densities (for rows 1+)
    density_values = np.nan_to_num(density_values, nan=0, posinf=0, neginf=0)
    density_values = np.clip(density_values, 0, None)  # Ensure no negative values

    # Additional check for non-positive molar mass (for rows 1+)
    density_values[molar_mass_values <= 0] = 0

    # --- Assign computed values back to DataFrame (only from row 1) ---
    df1.loc[1:, 'Density'] = density_values

    df1.loc[1:,'Density'] = np.round(density_values/1000,8)

    #---------------------------------------------------------------------------------------------------------------
    # VAPOR VOLUME PROCESSING
    final_results = []
    for i, row in df1.iterrows():
        tank_calcs = {}
        total_volume = 0

        for tank_num in range(1, len(tank_ids)+1):
            if f'tank{tank_num}_liquid_vol' not in row:
                continue

            # Define vapor-specific columns
            vap_temp_col = f'Vap_temp_TK{tank_num}'
            press_col = f'Press_TK{tank_num}'

            liquid_volume = row[f'tank{tank_num}_liquid_vol']
            tank_capacity = row[f'_tank{tank_num}_capacity']  # Get the specific tank capacity

            # Vapor calculations
            vap_corr = (273+15)/(273+row[vap_temp_col])*(1.013+row[press_col])/1.013*0.6785
            vnet = tank_capacity - liquid_volume
            vnet_corr = vnet * vap_corr

            # Handle zero density case
            if row['Density'] == 0:
                vap_volume = 0
            else:
                vap_volume = vnet_corr / row['Density'] / 1000

            # Total tank volume
            tank_total_vol = liquid_volume + vap_volume
            total_volume += tank_total_vol

            tank_calcs.update({
                f'tank{tank_num}_vap_vol': vap_volume,
                f'tank{tank_num}_total_vol': tank_total_vol,
                # Carry forward existing values
                f'tank{tank_num}_level': row[f'tank{tank_num}_level'],
                f'tank{tank_num}_liquid_vol': liquid_volume
            })

        # Remove temporary capacity fields
        row_dict = {k:v for k,v in row.items() if not k.startswith('_')}
        final_results.append({**row_dict, **tank_calcs, 'total_volume': total_volume})            

    df2 = pd.DataFrame(final_results)

    # Keep the first row's ROB_cams as-is, compute the rest
    df2.loc[1:, 'ROB_cams'] = (df2.loc[1:, 'total_volume'] * df2.loc[1:, 'Density']).round(1)

    df2.loc[1:,'ROB_cal'] = (df2.loc[1:,'ROB_cal']).round(1)

    df2.loc[1:, 'Discrepancy'] = (df2.loc[1:,'ROB_cal'] - df2.loc[1:,'ROB_cams']).round(1)  

    return df2

# Define a global cache dictionary to store df2 results by ship_id
df2_cache = {}

# Define a function to update this cache
def update_df2_cache(df2, ship_id):
    df2_cache[ship_id] = df2

# Assign the caching function to the global variable used in your API
update_df2_global = update_df2_cache

# ========== HEALTH CHECK ENDPOINT ==========
from fastapi import Body, HTTPException
from fastapi.encoders import jsonable_encoder

@app.post("/calculate")
def calculate_and_cache(ship_id: str = Body(..., embed=True),
                        date: str = Body(..., description="Date in YYYY-MM-DD format", embed=True)):

    # Validate ship_id
    if ship_id not in available_ships:
        raise HTTPException(status_code=400, detail=f"Unknown ship_id: {ship_id}")

    tank_ids = available_ships[ship_id]

    # Run full calculation pipeline
    df2 = calculate_columnwise(None, ship_id, tank_ids)  # df is fetched inside the function

    # Cache result
    update_df2_global(df2, ship_id)

    # Parse and validate date
    try:
        selected_date = pd.to_datetime(date).date()
    except Exception:
        raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format")

    # Convert and filter by date
    df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')
    output_df = df2[df2['Date'].dt.date == selected_date]

    if output_df.empty:
        raise HTTPException(status_code=404, detail="No record found for this date")

    # Clean for JSON serialization
    output_df = output_df.replace([np.nan, np.inf, -np.inf], None)
    output_df = output_df.applymap(
        lambda x: x.item() if isinstance(x, (np.generic, np.ndarray)) and getattr(x, 'ndim', 1) == 0 else x
    )
    output_df['Date'] = output_df['Date'].dt.strftime('%Y-%m-%d')

    selected_fields = ["Date", "ship_id", "ROB_cal", "ROB_cams", "Discrepancy", "PKI", "MN",
                       "Min_MN", "Molar_Mass", "Density", "tank1_liquid_vol","tank1_vap_vol", 
                       "tank1_total_vol","tank2_liquid_vol","tank2_vap_vol", "tank2_total_vol", 
                       "total_volume"]  # example fields
    
    # Define columns to round
    round_columns = [
        "PKI", "Molar_Mass",
        "tank1_liquid_vol", "tank1_vap_vol", "tank1_total_vol",
        "tank2_liquid_vol", "tank2_vap_vol", "tank2_total_vol",
        "total_volume"
    ]

    # Ensure all selected fields exist in the DataFrame
    missing_fields = [col for col in selected_fields if col not in output_df.columns]
    if missing_fields:
        raise HTTPException(status_code=500, detail=f"Missing columns in output: {missing_fields}")

    # Filter to selected fields
    output_df = output_df[selected_fields]

    # Round specified columns to 2 decimal places
    for col in round_columns:
        if col in output_df.columns:
            output_df[col] = output_df[col].round(2)

    return output_df.to_dict(orient="records")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "api": "CONSOLIDATED PKI MN Calculator",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
def root():
    """Root endpoint for basic info"""
    return JSONResponse(content={
        "message": "Welcome to the PKI MN Calculator API",
        "endpoints": ["/health", "/calculate", "/docs"],
        "version": "3.0.0"
    })    

# ========== IF NEEDED: Main Entrypoint ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)


# In[ ]:




